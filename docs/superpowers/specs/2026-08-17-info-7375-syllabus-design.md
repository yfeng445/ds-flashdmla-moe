# INFO 7375 Syllabus Documentation Design

## Goal

Add the public INFO 7375 *High Performance Computing for AI* syllabus to the
repository in a form that matches the existing Chinese textbook-style
documentation while also preserving an English-only reference version.

Source page:
<https://distinct-capricorn-c04.notion.site/Syllabus-High-Performance-Computing-for-AI-1f788315b6b48050946ede67ec5c086f>

## Files

- `docs/course/info-7375-syllabus.md`: the primary Chinese version.
- `docs/course/source/info-7375-syllabus.en.md`: an English-only structured
  reference version.
- `docs/index.md`: add a course-background entry linking to the Chinese version.

## Content Structure

Both syllabus documents will preserve the source hierarchy:

1. course description and learning outcomes;
2. prerequisites and preparatory reading;
3. textbooks;
4. course approach and representative papers;
5. grading;
6. the three-part, thirteen-week schedule;
7. assignment repository and submission rules.

The Chinese document will translate and lightly edit the material for clarity
and consistency with the rest of `docs/`. The English document will contain the
same information without Chinese commentary. It will be a faithful structured
restatement rather than a dump of Notion page markup.

## Source and Link Handling

- Record the source URL and snapshot date at the top of both documents.
- Preserve stable public links to textbooks, Stanford SLP chapters, and papers.
- Do not preserve Notion navigation controls, export-tool links, or other UI
  artifacts.
- Replace the expired signed Notion download URL for *The C Programming
  Language* with a bibliographic reference rather than a broken link.
- Keep source-specific administrative details when they explain the original
  course workflow, but label the documents as a historical syllabus snapshot so
  they are not mistaken for current repository requirements.

## Validation

- Confirm every source section appears in both versions.
- Confirm both documents contain the same thirteen-week schedule and grading
  breakdown.
- Check all retained Markdown links for valid syntax and reject temporary signed
  URLs.
- Confirm `docs/index.md` links to the Chinese document using a relative path.
- Run the repository's Markdown formatter or lint checks when available.

## Non-goals

- Do not copy course assignments or PDFs into this repository.
- Do not rewrite the existing technical chapters around the syllabus.
- Do not present the syllabus policies as current Northeastern University policy.
- Do not add a Notion synchronization mechanism.
