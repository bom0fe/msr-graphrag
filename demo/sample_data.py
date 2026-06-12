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
            "Arrival",
            "Arrival is a 2016 science fiction film directed by Denis Villeneuve. "
            "The film is based on Story of Your Life by Ted Chiang.",
            True,
            "hotpot_demo_1",
        ),
        Passage(
            "Story of Your Life",
            "Story of Your Life is a science fiction novella by American writer Ted Chiang. "
            "It won the Nebula Award for Best Novella.",
            True,
            "hotpot_demo_1",
        ),
        Passage(
            "Ted Chiang",
            "Ted Chiang is an American science fiction writer. His novella Story of Your "
            "Life was adapted into the film Arrival.",
            True,
            "hotpot_demo_1",
        ),
        Passage(
            "Denis Villeneuve",
            "Denis Villeneuve is a Canadian filmmaker who directed Arrival, Blade Runner "
            "2049, Dune, and other films.",
            False,
            "hotpot_demo_1",
        ),
        Passage(
            "Nebula Award for Best Novella",
            "The Nebula Award for Best Novella is presented by the Science Fiction and "
            "Fantasy Writers Association.",
            False,
            "hotpot_demo_1",
        ),
    ]
    q2_passages = [
        Passage(
            "The Social Network",
            "The Social Network is a 2010 biographical drama film directed by David "
            "Fincher. The screenplay was written by Aaron Sorkin.",
            True,
            "hotpot_demo_2",
        ),
        Passage(
            "David Fincher",
            "David Fincher directed The Social Network. He also directed Gone Girl, "
            "Zodiac, Fight Club, and Se7en.",
            True,
            "hotpot_demo_2",
        ),
        Passage(
            "Fight Club",
            "Fight Club is a 1999 film directed by David Fincher. It is based on the "
            "1996 novel Fight Club by Chuck Palahniuk.",
            True,
            "hotpot_demo_2",
        ),
        Passage(
            "Chuck Palahniuk",
            "Chuck Palahniuk is an American novelist who wrote the novel Fight Club.",
            True,
            "hotpot_demo_2",
        ),
        Passage(
            "Aaron Sorkin",
            "Aaron Sorkin is an American screenwriter who wrote the screenplay for "
            "The Social Network.",
            False,
            "hotpot_demo_2",
        ),
    ]
    return [
        QAExample(
            "hotpot_demo_1",
            "Who wrote the novella that the film Arrival is based on?",
            "Ted Chiang",
            gold_titles=["Arrival", "Story of Your Life", "Ted Chiang"],
            passages=q1_passages,
            num_hops=3,
            qtype="bridge",
            dataset="hotpotqa",
        ),
        QAExample(
            "hotpot_demo_2",
            "Who wrote the novel that another David Fincher film, Fight Club, was based on?",
            "Chuck Palahniuk",
            gold_titles=["The Social Network", "David Fincher", "Fight Club", "Chuck Palahniuk"],
            passages=q2_passages,
            num_hops=4,
            qtype="bridge",
            dataset="hotpotqa",
        ),
    ]


def _two_wiki_examples() -> List[QAExample]:
    q1_passages = [
        Passage(
            "Blade Runner 2049",
            "Blade Runner 2049 is a 2017 science fiction film directed by Denis "
            "Villeneuve. It is a sequel to Blade Runner.",
            True,
            "2wiki_demo_1",
        ),
        Passage(
            "Blade Runner",
            "Blade Runner is a 1982 science fiction film directed by Ridley Scott. "
            "It is loosely based on Philip K. Dick's novel Do Androids Dream of Electric Sheep?",
            True,
            "2wiki_demo_1",
        ),
        Passage(
            "Do Androids Dream of Electric Sheep?",
            "Do Androids Dream of Electric Sheep? is a 1968 science fiction novel by "
            "Philip K. Dick.",
            True,
            "2wiki_demo_1",
        ),
        Passage(
            "Philip K. Dick",
            "Philip K. Dick was an American science fiction writer whose works inspired "
            "films including Blade Runner, Total Recall, and Minority Report.",
            True,
            "2wiki_demo_1",
        ),
        Passage(
            "Denis Villeneuve",
            "Denis Villeneuve directed Arrival, Blade Runner 2049, Dune, and Sicario.",
            False,
            "2wiki_demo_1",
        ),
    ]
    q2_passages = [
        Passage(
            "The Imitation Game",
            "The Imitation Game is a 2014 biographical drama film about Alan Turing. "
            "It stars Benedict Cumberbatch as Alan Turing.",
            True,
            "2wiki_demo_2",
        ),
        Passage(
            "Alan Turing",
            "Alan Turing was an English mathematician and computer scientist. He worked "
            "at Bletchley Park during World War II.",
            True,
            "2wiki_demo_2",
        ),
        Passage(
            "Bletchley Park",
            "Bletchley Park was the principal centre of Allied code-breaking during "
            "World War II. It is located in Milton Keynes, England.",
            True,
            "2wiki_demo_2",
        ),
        Passage(
            "Milton Keynes",
            "Milton Keynes is a city and unitary authority area in Buckinghamshire, England.",
            True,
            "2wiki_demo_2",
        ),
        Passage(
            "Benedict Cumberbatch",
            "Benedict Cumberbatch portrayed Alan Turing in The Imitation Game.",
            False,
            "2wiki_demo_2",
        ),
    ]
    return [
        QAExample(
            "2wiki_demo_1",
            "Who wrote the novel that inspired the predecessor of Blade Runner 2049?",
            "Philip K. Dick",
            gold_titles=[
                "Blade Runner 2049",
                "Blade Runner",
                "Do Androids Dream of Electric Sheep?",
                "Philip K. Dick",
            ],
            passages=q1_passages,
            num_hops=4,
            qtype="bridge",
            dataset="2wiki",
        ),
        QAExample(
            "2wiki_demo_2",
            "In which city is the code-breaking centre where Alan Turing worked located?",
            "Milton Keynes",
            gold_titles=["The Imitation Game", "Alan Turing", "Bletchley Park", "Milton Keynes"],
            passages=q2_passages,
            num_hops=4,
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
