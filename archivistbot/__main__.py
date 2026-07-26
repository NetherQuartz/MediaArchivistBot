import asyncio
import logging

from .bot import BotApplication
from .config import get_settings


def configure_logging() -> None:
    level_name = get_settings().logging_level.upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    for noisy_logger in ("aiohttp", "httpcore", "httpx", "openai", "urllib3"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)


async def main() -> None:
    configure_logging()
    await BotApplication().run()


if __name__ == "__main__":
    asyncio.run(main())
