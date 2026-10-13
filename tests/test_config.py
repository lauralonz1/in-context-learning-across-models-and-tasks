from src.utils.config import load_config


def main():

    config = load_config()

    print("=" * 70)
    print("EXPERIMENT CONFIG TEST")
    print("=" * 70)

    print("\nProject:")
    print(config["project"])

    print("\nMMLU-Pro conditions:")

    for condition in config["benchmarks"]["mmlu_pro"]["conditions"]:
        print(condition)

    print("\nQuALITY conditions:")

    for condition in config["benchmarks"]["quality"]["conditions"]:
        print(condition)

    print("\nRandom seeds:")
    print(
        config["random_selection"]["seeds"]
    )

    assert len(
        config["benchmarks"]["mmlu_pro"]["conditions"]
    ) == 7

    assert len(
        config["benchmarks"]["quality"]["conditions"]
    ) == 5

    assert (
        config["quality"]["max_prompt_tokens"]
        == 32000
    )

    print("\n" + "=" * 70)
    print("CONFIG TEST PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()