import re
from urllib.parse import urlparse

from app.config import get_llm
from app.state import ResearchState

WRITER_PROMPT = """You are an expert technical writer.
Write a clear, well-structured research report in Markdown using ONLY the research notes provided.

Structure:
# <Title>
## Summary (3-4 sentences)
## Key Findings (use ### sub-headings as needed)
## Conclusion

Do NOT write a References section. It is added automatically from the numbered sources.

Citations:
- Cite web sources with numbers like [1] or [2, 3], using ONLY the numbered sources provided.
- Cite the user's documents with their source number AND page, like [4, p. 282].
  Research notes label document facts as [DOC: filename, p. N]: use the number of that
  document in the numbered sources, and the page from the label.
- Never add facts that are not in the research notes.

Length: aim for 400-700 words.

Images:
- If a list of available images is provided, embed the 1-3 most relevant ones.
  Prefer diagrams, architecture figures and charts over photos, logos or screenshots.
- Place each image inside the section it illustrates, in this exact format:
  ![Short caption](EXACT_URL)
  *Figure N: One-sentence caption. (Image source: DOMAIN)*
- Use ONLY the exact URLs from the list. Never invent or modify an image URL.
- If no image clearly matches the content, include no images."""

IMAGE_MARKDOWN = re.compile(r"!\[[^\]]*\]\(([^)\s]+)\)")
DOC_TAG = re.compile(r"\[DOC:\s*(.+?)(?:,\s*(pp?\.\s*[^\]]+))?\]")
# Matches [1] or [2, 3]  OR  [4, p. 282]
CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]|\[(\d+),\s*(pp?\.\s*[\d\s,–-]+?)\]")
REFERENCES_SECTION = re.compile(r"\n#+\s*References\b.*", re.DOTALL | re.IGNORECASE)


def _number_sources(state: ResearchState) -> list[str]:
    """All real sources: web URLs first, then documents with their pages grouped."""
    web = list(dict.fromkeys(state.get("sources", [])))

    pages_by_file: dict[str, list[int]] = {}
    for src in state.get("doc_sources", []):        # e.g. "book.pdf, p. 21"
        name, _, page = src.partition(", p. ")
        pages_by_file.setdefault(name, [])
        if page.isdigit():
            pages_by_file[name].append(int(page))

    docs = []
    for name, pages in pages_by_file.items():
        pages = sorted(set(pages))
        if not pages:
            docs.append(name)
        elif len(pages) == 1:
            docs.append(f"{name}, p. {pages[0]}")
        else:
            docs.append(f"{name}, pp. {', '.join(map(str, pages))}")
    return web + docs


def _convert_doc_tags(body: str, numbered_sources: list[str]) -> str:
    """Turn leftover [DOC: book.pdf, p. 282] tags into [n, p. 282] (n = that document's number)."""
    number_of = {}
    for i, src in enumerate(numbered_sources, start=1):
        if not src.startswith("http"):
            number_of[src.split(", p")[0]] = i   # "book.pdf, pp. 281, 282" -> "book.pdf"

    def convert(match):
        n = number_of.get(match.group(1).strip())
        if n is None:
            return ""  # a document we have no record of: drop the tag
        return f"[{n}, {match.group(2).strip()}]" if match.group(2) else f"[{n}]"

    return DOC_TAG.sub(convert, body)


def _finalize_references(draft: str, numbered_sources: list[str]) -> str:
    """Code guardrail: fix citations, renumber in order of use, append the References list."""
    body = REFERENCES_SECTION.sub("", draft).rstrip()   # remove any list the LLM wrote anyway
    body = _convert_doc_tags(body, numbered_sources)

    def cited_numbers(match) -> list[int]:
        if match.group(1):                               # [1] or [2, 3]
            return [int(n) for n in match.group(1).split(",")]
        return [int(match.group(2))]                     # [4, p. 282]

    order: list[int] = []
    for match in CITATION.finditer(body):
        for n in cited_numbers(match):
            if 1 <= n <= len(numbered_sources) and n not in order:
                order.append(n)
    new_number = {old: new for new, old in enumerate(order, start=1)}

    def renumber(match):
        nums = [new_number[n] for n in cited_numbers(match) if n in new_number]
        if not nums:
            return ""                                    # invalid citation: remove it
        if match.group(1):
            return "[" + ", ".join(map(str, nums)) + "]"
        return f"[{nums[0]}, {match.group(3).strip()}]"  # keep the page number

    body = CITATION.sub(renumber, body)
    if not order:
        return body

    references = "\n".join(f"{new}. {numbered_sources[old - 1]}" for old, new in new_number.items())
    return f"{body}\n\n## References\n{references}"

def _format_images(images: list[dict]) -> str:
    lines = []
    for i, img in enumerate(images, start=1):
        lines.append(
            f"[{i}] URL: {img['url']}\n"
            f"    Domain: {urlparse(img['url']).netloc}\n"
            f"    Description: {img['description'] or 'no description'}"
        )
    return "\n".join(lines)


def _remove_unknown_images(draft: str, allowed_urls: set[str]) -> str:
    return IMAGE_MARKDOWN.sub(
        lambda m: m.group(0) if m.group(1) in allowed_urls else "", draft
    )


def writer_node(state: ResearchState) -> dict:
    revision = state.get("revision_count", 0)
    is_revision = bool(state.get("draft")) and bool(state.get("critique"))
    print(f"✍️  Writer: {'revising' if is_revision else 'writing'} draft #{revision + 1}...")

    images = list({img["url"]: img for img in state.get("images", [])}.values())
    numbered = _number_sources(state)
    sources_text = "\n".join(f"[{i}] {s}" for i, s in enumerate(numbered, start=1)) or "none"

    message = (
        f"Topic: {state['topic']}\n\n"
        f"Research notes:\n{chr(10).join(state['research_notes'])}\n\n"
        f"Numbered sources (cite ONLY these numbers):\n{sources_text}\n\n"
        f"Available images:\n{_format_images(images) if images else 'none'}"
    )
    if is_revision:
        message += (
            f"\n\nYour previous draft:\n{state['draft']}\n\n"
            f"Reviewer feedback to address:\n{state['critique']}\n\n"
            "Rewrite the report so it fully addresses the feedback."
        )

    response = get_llm().invoke([("system", WRITER_PROMPT), ("human", message)])

    draft = _remove_unknown_images(response.content, {img["url"] for img in images}) #type: ignore
    draft = _finalize_references(draft, numbered)
    print(f"\n   🖼  Embedded {len(IMAGE_MARKDOWN.findall(draft))} image(s).")

    return {"draft": draft, "revision_count": revision + 1, "last_agent": "writer"}