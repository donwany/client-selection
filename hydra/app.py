import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base="1.1", config_path="./configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    print(cfg.model.tokenizer)
    print(cfg.processing.batch_size)
    print(cfg.processing.max_length)
    print(cfg.training.log_every_n_steps)
    print(cfg.training.deterministic)
    print(cfg.training.limit_train_batches)
    print(cfg.training.limit_val_batches)


if __name__ == "__main__":
    main()
