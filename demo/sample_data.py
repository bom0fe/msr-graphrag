"""Small built-in demo samples for each supported benchmark dataset."""
from __future__ import annotations

from typing import List, Tuple

from msr_graphrag.data.schema import Corpus, Passage, QAExample
from msr_graphrag.data.loaders import build_corpus


def load_demo_dataset(name: str) -> Tuple[List[QAExample], Corpus]:
    """Return two hand-curated examples for a demo dataset."""
    key = name.lower()
    if key == "hotpotqa":
        examples = _hotpotqa_examples()
    elif key == "2wiki":
        examples = _two_wiki_examples()
    elif key == "musique":
        examples = _musique_examples()
    else:
        raise ValueError(f"unknown built-in demo dataset: {name}")
    return examples, build_corpus(examples, name=f"{key}_demo")


def _hotpotqa_examples() -> List[QAExample]:
    q1_passages = [
        Passage(
            "Inception",
            "Inception is a 2010 science fiction action film written and directed by "
            "Christopher Nolan. The film stars Leonardo DiCaprio as a professional thief.",
            True,
            "hotpot_demo_1",
        ),
        Passage(
            "Christopher Nolan",
            "Christopher Nolan is a British-American filmmaker. He directed films such as "
            "Memento, The Dark Knight, Inception, Interstellar, and Oppenheimer.",
            True,
            "hotpot_demo_1",
        ),
        Passage(
            "Leonardo DiCaprio",
            "Leonardo DiCaprio is an American actor who starred in Titanic, Inception, "
            "The Revenant, and The Wolf of Wall Street.",
            False,
            "hotpot_demo_1",
        ),
        Passage(
            "Interstellar",
            "Interstellar is a 2014 science fiction film directed by Christopher Nolan.",
            False,
            "hotpot_demo_1",
        ),
    ]
    q2_passages = [
        Passage(
            "Pride and Prejudice",
            "Pride and Prejudice is an 1813 novel of manners written by Jane Austen.",
            True,
            "hotpot_demo_2",
        ),
        Passage(
            "Jane Austen",
            "Jane Austen was an English novelist known for Sense and Sensibility, "
            "Pride and Prejudice, Mansfield Park, Emma, and Persuasion.",
            True,
            "hotpot_demo_2",
        ),
        Passage(
            "Emma",
            "Emma is a novel by Jane Austen, first published in 1815.",
            False,
            "hotpot_demo_2",
        ),
        Passage(
            "Charles Dickens",
            "Charles Dickens was an English writer known for Oliver Twist and Great Expectations.",
            False,
            "hotpot_demo_2",
        ),
    ]
    return [
        QAExample(
            "hotpot_demo_1",
            "Who directed the film Inception?",
            "Christopher Nolan",
            gold_titles=["Inception", "Christopher Nolan"],
            passages=q1_passages,
            num_hops=2,
            qtype="bridge",
            dataset="hotpotqa",
        ),
        QAExample(
            "hotpot_demo_2",
            "Which author wrote the novel Pride and Prejudice?",
            "Jane Austen",
            gold_titles=["Pride and Prejudice", "Jane Austen"],
            passages=q2_passages,
            num_hops=2,
            qtype="bridge",
            dataset="hotpotqa",
        ),
    ]


def _two_wiki_examples() -> List[QAExample]:
    q1_passages = [
        Passage(
            "The Martian",
            "The Martian is a 2015 science fiction film directed by Ridley Scott. "
            "It is based on Andy Weir's novel of the same name.",
            True,
            "2wiki_demo_1",
        ),
        Passage(
            "Ridley Scott",
            "Ridley Scott is an English film director and producer. He directed Alien, "
            "Blade Runner, Gladiator, The Martian, and other films.",
            True,
            "2wiki_demo_1",
        ),
        Passage(
            "Andy Weir",
            "Andy Weir is an American novelist known for The Martian and Project Hail Mary.",
            False,
            "2wiki_demo_1",
        ),
        Passage(
            "Gladiator",
            "Gladiator is a historical epic film directed by Ridley Scott.",
            False,
            "2wiki_demo_1",
        ),
    ]
    q2_passages = [
        Passage(
            "The Lord of the Rings",
            "The Lord of the Rings is an epic high-fantasy novel by J. R. R. Tolkien.",
            True,
            "2wiki_demo_2",
        ),
        Passage(
            "J. R. R. Tolkien",
            "J. R. R. Tolkien was an English writer, philologist, and academic. "
            "He authored The Hobbit and The Lord of the Rings.",
            True,
            "2wiki_demo_2",
        ),
        Passage(
            "The Hobbit",
            "The Hobbit is a children's fantasy novel by J. R. R. Tolkien.",
            False,
            "2wiki_demo_2",
        ),
        Passage(
            "C. S. Lewis",
            "C. S. Lewis was a British writer known for The Chronicles of Narnia.",
            False,
            "2wiki_demo_2",
        ),
    ]
    return [
        QAExample(
            "2wiki_demo_1",
            "Who directed the film adaptation of Andy Weir's novel The Martian?",
            "Ridley Scott",
            gold_titles=["The Martian", "Ridley Scott"],
            passages=q1_passages,
            num_hops=2,
            qtype="bridge",
            dataset="2wiki",
        ),
        QAExample(
            "2wiki_demo_2",
            "Who wrote the epic fantasy novel The Lord of the Rings?",
            "J. R. R. Tolkien",
            answer_aliases=["Tolkien"],
            gold_titles=["The Lord of the Rings", "J. R. R. Tolkien"],
            passages=q2_passages,
            num_hops=2,
            qtype="bridge",
            dataset="2wiki",
        ),
    ]


def _musique_examples() -> List[QAExample]:
    q1_passages = [
        Passage(
            "The Theory of Relativity",
            "The theory of relativity is a scientific theory developed by Albert Einstein.",
            True,
            "musique_demo_1",
        ),
        Passage(
            "Albert Einstein",
            "Albert Einstein was a theoretical physicist born in Ulm, in the Kingdom of "
            "Wuerttemberg in the German Empire.",
            True,
            "musique_demo_1",
        ),
        Passage(
            "Ulm",
            "Ulm is a city in the German state of Baden-Wuerttemberg.",
            True,
            "musique_demo_1",
        ),
        Passage(
            "Isaac Newton",
            "Isaac Newton was an English mathematician and physicist.",
            False,
            "musique_demo_1",
        ),
    ]
    q2_passages = [
        Passage(
            "The Great Gatsby",
            "The Great Gatsby is a 1925 novel by American writer F. Scott Fitzgerald.",
            True,
            "musique_demo_2",
        ),
        Passage(
            "F. Scott Fitzgerald",
            "F. Scott Fitzgerald was an American novelist associated with the Jazz Age.",
            True,
            "musique_demo_2",
        ),
        Passage(
            "Jazz Age",
            "The Jazz Age was a period in the 1920s marked by jazz music, social change, "
            "and modernist culture.",
            True,
            "musique_demo_2",
        ),
        Passage(
            "Ernest Hemingway",
            "Ernest Hemingway was an American novelist and journalist.",
            False,
            "musique_demo_2",
        ),
    ]
    return [
        QAExample(
            "musique_demo_1",
            "In which city was the physicist who developed relativity born?",
            "Ulm",
            answer_aliases=["Ulm, Germany"],
            gold_titles=["The Theory of Relativity", "Albert Einstein", "Ulm"],
            passages=q1_passages,
            num_hops=3,
            qtype="bridge",
            dataset="musique",
        ),
        QAExample(
            "musique_demo_2",
            "Which cultural period is associated with the author of The Great Gatsby?",
            "Jazz Age",
            gold_titles=["The Great Gatsby", "F. Scott Fitzgerald", "Jazz Age"],
            passages=q2_passages,
            num_hops=3,
            qtype="bridge",
            dataset="musique",
        ),
    ]
