from unittest.mock import MagicMock, patch

import pytest

from main import LoraTrainRequest


def test_fine_tune_lora_endpoint(client):
    """
    Test that the /fine-tune-lora endpoint validates input,
    queues the task, and returns the correct response.
    """
    payload = {
        "text_content": "query\tpassage",
        "model_name": "test_lora_model",
        "r": 8,
        "lora_alpha": 16,
        "use_qlora": True,
    }

    # We mock train_lora_worker so it doesn't actually run
    with patch("main.train_lora_worker") as mock_worker:
        # We also need to mock detect_model_profile to pass validation
        with patch("main.detect_model_profile") as mock_detect:
            mock_detect.return_value.requires_prefix = True  # dummy

            response = client.post("/fine-tune-lora", json=payload)

            assert response.status_code == 200
            data = response.json()
            assert "job_id" in data
            assert data["status"] == "pending"
            assert data["type"] == "lora"

            # Verify worker was called
            mock_worker.assert_called_once()
            call_args = mock_worker.call_args
            assert call_args[0][0] == data["job_id"]  # job_id
            req_obj = call_args[0][1]
            assert isinstance(req_obj, LoraTrainRequest)
            assert req_obj.r == 8
            assert req_obj.use_qlora is True


def test_fine_tune_lora_validation_error(client):
    # Missing required fields
    response = client.post("/fine-tune-lora", json={"model_name": "foo"})
    assert response.status_code == 422


def test_train_lora_worker_flow(mock_sentence_transformer):
    """
    Test the flow of train_lora_worker with heavy mocking.
    We check if PEFT and Quantization configs are initialized correctly.
    """
    from main import jobs, train_lora_worker

    # Mock all the heavy imports and internal calls
    with (
        patch("main.BitsAndBytesConfig") as MockBnBConfig,
        patch("main.LoraConfig") as MockLoraConfig,
        patch("main.get_peft_model") as _mock_get_peft,
        patch("main.prepare_model_for_kbit_training") as mock_prep_kbit,
        patch("main.SentenceTransformerTrainingArguments") as _MockArgs,
        patch("main.SentenceTransformerTrainer") as MockTrainer,
        patch("main.detect_model_profile") as mock_detect,
        patch("main.datasets.Dataset.from_dict") as _mock_dataset,
    ):

        # Setup mocks
        mock_detect.return_value.query_prefix = "q: "
        mock_detect.return_value.passage_prefix = "p: "
        mock_detect.return_value.max_seq_length = 512

        # Internal model mocks
        mock_stm = mock_sentence_transformer
        mock_stm._first_module.return_value.auto_model = (
            MagicMock()
        )  # The internal HF model

        req = LoraTrainRequest(
            text_content="A\tB", model_name="my_lora", r=8, use_qlora=True
        )

        job_id = "job_123"
        jobs[job_id] = {}

        # Run worker
        train_lora_worker(job_id, req)

        # Assertions

        # 1. Q-LoRA Config used?
        MockBnBConfig.assert_called_with(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=pytest.importorskip("torch").float16,
        )

        # 2. Model prepared for kbit?
        mock_prep_kbit.assert_called_once()

        # 3. Lora Config created?
        MockLoraConfig.assert_called_with(
            r=8,
            lora_alpha=32,  # default
            lora_dropout=0.05,  # default
            bias="none",
            task_type="FEATURE_EXTRACTION",
            target_modules=["query", "key", "value", "dense"],
        )

        # 4. Trainer initialized?
        MockTrainer.assert_called_once()
        trainer_instance = MockTrainer.return_value
        trainer_instance.train.assert_called_once()

        # 5. Model saved?
        # With LoRA, we verify save is called.
        mock_stm.save.assert_called()

        # 6. Job completed
        assert jobs[job_id]["status"] == "completed"
