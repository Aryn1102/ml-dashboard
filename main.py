from core.ingestion import load_data
from services.trainer import train_model
from config import DATA_PATH
from utils.logger import setup_logger

logger = setup_logger()


def main():
    try:
        df=load_data(DATA_PATH)
        logger.info("Starting training service")

        metrics = train_model(df)

        logger.info(f"Training completed: {metrics}")

    except Exception as e:
        logger.exception(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()