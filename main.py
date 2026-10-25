"""Repository entry point for the Wingman coach."""


def main(argv=None) -> int:
    print("Hello from Wingman!")
    from src.main import main as run_main

    return run_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
