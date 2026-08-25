# HPC for AI Notion Notes Documentation Design

## Goal

Turn the public *HPC for AI* Notion collection into maintainable Chinese course
notes with a parallel English-only source layer. Reframe the existing AI Infra
interview guide as a general knowledge and career guide rather than a pitch for
this repository as an interview project.

Parent source:
<https://distinct-capricorn-c04.notion.site/HPC-for-AI-20d88315b6b480538083fbe724df2902>

## Source Scope

The collection contains the previously imported syllabus and eleven weekly
pages:

1. Parallelism;
2. CUDA;
3. Memory;
4. Attention;
5. Layout Algebra;
6. CuTe GEMM;
7. Communication;
8. Data and Expert Parallelism;
9. Sequence Parallelism;
10. Inference Systems;
11. Career Building.

The syllabus remains in `docs/course/info-7375-syllabus.md` and will be linked
from the new collection index rather than duplicated.

## File Structure

Create a one-file-per-week mirror:

```text
docs/course/
|-- hpc-for-ai.md
|-- notes/
|   |-- 01-parallelism.md
|   |-- 02-cuda.md
|   |-- 03-memory.md
|   |-- 04-attention.md
|   |-- 05-layout-algebra.md
|   |-- 06-cute-gemm.md
|   |-- 07-communication.md
|   |-- 08-data-expert-parallelism.md
|   |-- 09-sequence-parallelism.md
|   |-- 10-inference-systems.md
|   `-- 11-career-building.md
`-- source/
    |-- hpc-for-ai.en.md
    `-- notes/
        |-- 01-parallelism.en.md
        |-- 02-cuda.en.md
        |-- 03-memory.en.md
        |-- 04-attention.en.md
        |-- 05-layout-algebra.en.md
        |-- 06-cute-gemm.en.md
        |-- 07-communication.en.md
        |-- 08-data-expert-parallelism.en.md
        |-- 09-sequence-parallelism.en.md
        |-- 10-inference-systems.en.md
        `-- 11-career-building.en.md
```

Also modify:

- `docs/index.md` to link the Chinese collection index and describe this
  repository as an AI Infra knowledge collection;
- `docs/infra-interview-guide.md` to remove project-pitch framing and include a
  complete, interview-oriented version of Week 11.

`docs/infra-mock-interview.md` is outside this change.

## Content Model

Each weekly file will:

- record the exact source page and snapshot date;
- retain the source's meaningful heading hierarchy and topic coverage;
- use concise explanations, formulas, tables, and short examples appropriate to
  the topic;
- preserve stable links to papers, books, documentation, and code;
- remove Notion navigation, export controls, signed downloads, and other UI
  artifacts;
- avoid presenting old course administration as current policy.

The Chinese files are the primary reader-facing notes and may add short
transitions that connect a week to this repository's existing chapters. The
English files contain the same factual coverage without Chinese commentary.
Both layers are faithful structured restatements, not raw Notion markup dumps.
Long source passages and large source code listings will be summarized; only
short fragments needed to explain a technical point will be retained.

If a public child page cannot be read, its task stops rather than inventing
content. The source URL and failure must be reported before implementation is
considered complete.

## Collection Indexes

`docs/course/hpc-for-ai.md` will provide:

- the collection source and snapshot date;
- a link to the existing syllabus;
- an ordered table of all eleven Chinese weekly notes;
- a short description of each week's role in the learning path;
- a link to the English collection index.

`docs/course/source/hpc-for-ai.en.md` will mirror this navigation in English and
link each English weekly page back to its Chinese counterpart.

## AI Infra Guide Reframing

`docs/infra-interview-guide.md` will become a general AI Infra learning and
interview guide. The rewrite will:

- remove the project elevator pitches, candidate positioning, personal
  contribution language, repository performance claims, and instructions for
  presenting this repository as interview experience;
- retain reusable explanations of Softmax, FlashAttention, MLA, MoE, expert
  parallelism, GPU memory, distributed training, C++, hand-written exercises,
  and performance-evidence methodology;
- express implementation-specific material as general validation patterns or
  study examples, not as a candidate project claim;
- add the full Career Building material in an interview-oriented organization:
  role selection, skill map, evidence-building, resume principles, preparation
  loop, behavioral communication, and questions for the interviewer;
- link to `notes/11-career-building.md` as the course-order version of the same
  source material.

The Week 11 course note and the guide both preserve the complete topic coverage.
They are deliberately organized differently and each records the same Notion
source, so the duplication is explicit rather than accidental.

## Validation

- Confirm all eleven Chinese weekly files and all eleven English counterparts
  exist and have matching numeric prefixes.
- Confirm every weekly file records one source URL and the snapshot date.
- Compare heading/topic checklists between each Chinese-English pair.
- Confirm the Chinese and English collection indexes link all eleven weeks and
  the existing syllabus.
- Confirm English-only files contain no Chinese prose.
- Reject `file.notion.so`, `notiontopdf`, temporary signatures, and Notion UI
  query parameters in the imported notes.
- Confirm `docs/index.md` no longer describes the interview material as personal
  project contribution evidence.
- Confirm `docs/infra-interview-guide.md` contains no project pitch or direction
  to use this repository as interview experience.
- Run Markdown checks and the repository test suite before final delivery.

## Non-goals

- Do not copy the original course PDFs, books, assignments, or large code
  listings into this repository.
- Do not add automatic Notion synchronization.
- Do not rewrite supported Python or CUDA APIs as part of this documentation
  import.
- Do not modify `docs/infra-mock-interview.md` in this change.
- Do not claim that this repository itself is an interview project.
