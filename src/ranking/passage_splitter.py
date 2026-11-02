import hashlib
import re
import spacy


class PassageSplitter:
    def __init__(self, sentences_per_passage=3, overlap=1):
        if sentences_per_passage < 1:
            raise ValueError("sentences_per_passage must be positive")

        if overlap < 0 or overlap >= sentences_per_passage:
            raise ValueError(
                "overlap must be smaller than sentences_per_passage"
            )

        self.sentences_per_passage = sentences_per_passage
        self.overlap = overlap
        self.step = sentences_per_passage - overlap
        self.nlp = spacy.load("en_core_web_sm")

    def split_sentences(self, text):
        text = re.sub(r"\s+", " ", text).strip()

        if not text:
            return []

        doc = self.nlp(text)

        return [
            sentence.text.strip()
            for sentence in doc.sents
            if sentence.text.strip()
        ]

    def split_article(self, article):
        text = article.get("text", "")

        if not isinstance(text, str) or not text.strip():
            return []

        sentences = self.split_sentences(text)
        passages = []

        for start in range(0, len(sentences), self.step):
            selected = sentences[
                start:start + self.sentences_per_passage
            ]

            if not selected:
                continue

            passage_text = " ".join(selected)
            source_url = article.get("url", "")

            passage_id = hashlib.sha256(
                f"{source_url}|{passage_text}".encode("utf-8")
            ).hexdigest()[:16]

            passages.append({
                "passage_id": passage_id,
                "text": passage_text,
                "title": article.get("title", ""),
                "url": source_url,
                "source": article.get("source", ""),
                "published_date": article.get("published_date"),
                "start_sentence": start,
                "end_sentence": start + len(selected)
            })

        return passages

    def split_articles(self, articles):
        passages = []

        for article in articles:
            passages.extend(self.split_article(article))

        return passages