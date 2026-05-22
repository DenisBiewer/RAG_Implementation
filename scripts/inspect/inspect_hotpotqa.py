from datasets import load_dataset

""" This script is used to inspect the HotpotQA dataset """

def main():
    dataset = load_dataset("hotpot_qa", "fullwiki")

    print("Available splits:", dataset.keys())

    sample = dataset["validation"][0]

    print("\nSample keys:")
    print(sample.keys())

    print("\nQuestion:")
    print(sample["question"])

    print("\nAnswer:")
    print(sample["answer"])

    print("\nSupporting facts:")
    print(sample["supporting_facts"])

    print("\nContext example:")
    context = sample["context"]
    print("Keys:", context.keys())

    titles = context["title"][:2]
    sentences = context["sentences"][:2]

    for t, s in zip(titles, sentences):
        print("\nTitle:", t)
        print("Sentences:", s[:2])  # show first 2 sentences


if __name__ == "__main__":
    main()