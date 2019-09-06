local stringToBool(s) =
  if s == "true" then true
  else if s == "false" then false
  else error "invalid boolean: " + std.manifestJson(s);

local boolToInt(s) =
  if s == true then 1
  else if s == false then 0
  else error "invalid boolean: " + std.manifestJson(s);

{
  "random_seed": std.parseInt(std.extVar("SEED")),
  "pytorch_seed": std.parseInt(std.extVar("PYTORCH_SEED")),
  "numpy_seed": std.parseInt(std.extVar("NUMPY_SEED")),
  "dataset_reader":{
    "type":"SeqClassificationReader",
    "lazy": false,
    "sent_max_len": std.extVar("SENT_MAX_LEN"),
    "word_splitter": "bert-basic",
    "max_sent_per_example": std.extVar("MAX_SENT_PER_EXAMPLE"),
    "token_indexers": {
          "bert": {
              "type": "bert-pretrained",
              "pretrained_model": std.extVar("BERT_VOCAB"),
              "do_lowercase": true,
              "use_starting_offsets": false
          },
    },
    "use_sep": std.extVar("USE_SEP"),
    "sci_sum": stringToBool(std.extVar("SCI_SUM")),
    "use_abstract_scores": stringToBool(std.extVar("USE_ABSTRACT_SCORES")),
    "sci_sum_fake_scores": stringToBool(std.extVar("SCI_SUM_FAKE_SCORES")),
  },

  "train_data_path": std.extVar("TRAIN_PATH"),
  "validation_data_path": std.extVar("DEV_PATH"),
  "test_data_path": std.extVar("TEST_PATH"),
  "evaluate_on_test": true,
  "model": {
    "type": "SeqClassificationModel",
    "text_field_embedder": {
        "allow_unmatched_keys": true,
        "embedder_to_indexer_map": {
            "bert": if stringToBool(std.extVar("USE_SEP")) then ["bert"] else ["bert", "bert-offsets"],
            "tokens": ["tokens"],
        },
        "token_embedders": {
            "bert": {
                "type": "bert-pretrained",
                "pretrained_model": std.extVar("BERT_WEIGHTS"),
                "requires_grad": 'all',
                "top_layer_only": false,
            }
        }
    },
    "use_sep": std.extVar("USE_SEP"),
    "with_crf": std.extVar("WITH_CRF"),
    "bert_dropout": 0.1,
    "sci_sum": stringToBool(std.extVar("SCI_SUM")),
    "additional_feature_size": boolToInt(stringToBool(std.extVar("USE_ABSTRACT_SCORES"))),
    "self_attn": {
      "type": "stacked_self_attention",
      "input_dim": 768,
      "projection_dim": 100,
      "feedforward_hidden_dim": 50,
      "num_layers": 2,
      "num_attention_heads": 2,
      "hidden_dim": 100,
    },
  },
  "iterator": {
    "type": "bucket",
    "sorting_keys": [["sentences", "num_fields"]],
    "batch_size" : std.parseInt(std.extVar("BATCH_SIZE")),
    "cache_instances": true,
    "biggest_batch_first": true
  },

  "trainer": {
    "num_epochs": std.parseInt(std.extVar("NUM_EPOCHS")),
    "grad_clipping": 1.0,
    "patience": 5,
    "model_save_interval": 3600,
    "validation_metric": if stringToBool(std.extVar("SCI_SUM")) then "-loss" else '+acc',
    "min_delta": 0.001,
    "cuda_device": std.parseInt(std.extVar("cuda_device")),
    "gradient_accumulation_batch_size": 32,
    "optimizer": {
      "type": "bert_adam",
      "lr": std.extVar("LR"),
      "t_total": -1,
      "max_grad_norm": 1.0,
      "weight_decay": 0.01,
      "parameter_groups": [
        [["bias", "LayerNorm.bias", "LayerNorm.weight", "layer_norm.weight"], {"weight_decay": 0.0}],
      ],
    },
    "should_log_learning_rate": true,
    "learning_rate_scheduler": {
      "type": "slanted_triangular",
      "num_epochs": std.parseInt(std.extVar("NUM_EPOCHS")),
      "num_steps_per_epoch": std.parseInt(std.extVar("TRAINING_DATA_INSTANCES")) / 32,
      "cut_frac": 0.1,
    },
  }
}