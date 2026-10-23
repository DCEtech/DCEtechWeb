---
name: english-post-review
description: Use when writing, drafting or reviewing a blog post in content/*.md for danielcejudo.com. Corrects Daniel's English and enforces the post structure.
---

# English post review

Daniel writes posts in English to improve. The goal is that he learns, not only that the
post comes out clean.

## When reviewing
1. Correct the text, keeping his voice; do not rewrite it into generic prose.
2. After the corrected version, list each correction briefly: original → fixed → why
   (grammar, naturalness or technical vocabulary). Explanations in Spanish.
3. Watch for his recurring errors:
   - `´` instead of the apostrophe `'`
   - ordinals written as `1º` instead of `1st`
   - double letters: recommendation, professional, useful
   - past tenses

## Post format
- Front-matter complete: Title, Date, Slug, Tags, Summary and Description
  (≤160 chars; the theme prints `Description` as `<meta name="description">` in article.html).
- Headings with `##`; code blocks always with a language (```python).
- If the post includes code, link to the source repository.
- File name without spaces or special characters (e.g. `why-i-love-data.md`).
