"""Run the desktop assistant: python main.py."""

import logging
import sys

from app import App


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    application = App()
    return application.run()


if __name__ == "__main__":
    sys.exit(main())
