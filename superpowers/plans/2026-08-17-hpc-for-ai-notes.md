# HPC for AI Notion Notes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Import the eleven public HPC for AI weekly pages as Chinese and English Markdown notes, add bilingual collection indexes, and convert the AI Infra guide from a repository-project pitch into a general knowledge and career guide.

**Architecture:** Each Notion week becomes one Chinese reader-facing file and one English-only structured source file with matching numeric prefixes. Collection indexes provide navigation and link the existing syllabus; Week 11 also feeds a complete, differently organized career section in the general AI Infra guide.

**Tech Stack:** Public Notion pages, GitHub-flavored Markdown, browser-based read-only extraction, PowerShell validation commands, Git.

## Global Constraints

- Parent source: <https://distinct-capricorn-c04.notion.site/HPC-for-AI-20d88315b6b480538083fbe724df2902>.
- Snapshot date: `2026-08-17`.
- Create exactly eleven Chinese weekly files and eleven English counterparts with matching numeric prefixes.
- Every weekly file records its exact public source URL and snapshot date.
- English source files contain no Chinese prose; Chinese files are the primary reader-facing notes.
- Preserve topic coverage, meaningful hierarchy, formulas, short examples, and stable public references without dumping Notion markup verbatim.
- Do not retain `file.notion.so`, `notiontopdf`, signed download URLs, `pvs=` query parameters, or Notion UI artifacts.
- Do not copy books, papers, assignments, large code listings, or course PDFs into the repository.
- Do not modify Python or CUDA APIs.
- Do not modify `docs/infra-mock-interview.md`.
- Do not present this repository as an interview project or personal contribution claim.
- If a source page cannot be read after focused browser retries, stop that task and report the missing page rather than inventing content.

---

### Task 1: Weeks 1-4 Foundations

**Files:**
- Create: `docs/course/notes/01-parallelism.md`
- Create: `docs/course/notes/02-cuda.md`
- Create: `docs/course/notes/03-memory.md`
- Create: `docs/course/notes/04-attention.md`
- Create: `docs/course/source/notes/01-parallelism.en.md`
- Create: `docs/course/source/notes/02-cuda.en.md`
- Create: `docs/course/source/notes/03-memory.en.md`
- Create: `docs/course/source/notes/04-attention.en.md`

**Interfaces:**
- Consumes: the four public pages listed below.
- Produces: the first four Chinese-English weekly pairs for the collection indexes.

Source pages:

- Week 1: <https://distinct-capricorn-c04.notion.site/Week-1-Parallelism-26288315b6b4808583c0ecee574eca71>
- Week 2: <https://distinct-capricorn-c04.notion.site/Week-2-CUDA-26388315b6b480d480aec7e22cde5776>
- Week 3: <https://distinct-capricorn-c04.notion.site/Week-3-Memory-26388315b6b48016a19bc6f451f9e1eb>
- Week 4: <https://distinct-capricorn-c04.notion.site/Week-4-Attention-26a88315b6b480f1b26ffba505ff0677>

- [ ] **Step 1: Read each source page separately**

  Open one page at a time. Capture its rendered headings, lists, formulas,
  tables, short code examples, and stable outbound links. Ignore navigation,
  export buttons, and unrelated parent-page UI. Do not move to the next page
  until the current page title and main content are visible.

- [ ] **Step 2: Write the four Chinese notes**

  For every file, use a Chinese title, source metadata, the original weekly role,
  the complete topic hierarchy, concise technical explanations, and a final
  `延伸阅读` section for stable external links. Link each file to its English
  counterpart using `../source/notes/<matching-name>.en.md`.

- [ ] **Step 3: Write the four English notes**

  Mirror the same factual topic coverage and outbound links in English-only
  prose. Link each file back to `../../notes/<matching-name>.md`.

- [ ] **Step 4: Validate the first batch**

  Run:

  ```powershell
  $zh = Get-ChildItem docs/course/notes -File | Where-Object Name -Match '^0[1-4]-.*\.md$'
  $en = Get-ChildItem docs/course/source/notes -File | Where-Object Name -Match '^0[1-4]-.*\.en\.md$'
  "zh=$($zh.Count) en=$($en.Count)"
  rg -n -P '\p{Han}' docs/course/source/notes -g '0[1-4]-*.en.md'
  rg -n 'file\.notion\.so|notiontopdf|pvs=|signature=' docs/course/notes docs/course/source/notes
  git diff --check
  ```

  Expected: `zh=4 en=4`; the Chinese-character and forbidden-URL searches have
  no matches; `git diff --check` prints nothing.

- [ ] **Step 5: Commit Weeks 1-4**

  ```powershell
  git add -- docs/course/notes/0[1-4]-*.md docs/course/source/notes/0[1-4]-*.en.md
  git commit -m "Add HPC for AI foundation notes"
  ```

### Task 2: Weeks 5-7 Kernel Layout and Communication

**Files:**
- Create: `docs/course/notes/05-layout-algebra.md`
- Create: `docs/course/notes/06-cute-gemm.md`
- Create: `docs/course/notes/07-communication.md`
- Create: `docs/course/source/notes/05-layout-algebra.en.md`
- Create: `docs/course/source/notes/06-cute-gemm.en.md`
- Create: `docs/course/source/notes/07-communication.en.md`

**Interfaces:**
- Consumes: the three public pages below and the metadata/link conventions from Task 1.
- Produces: the kernel-layout and communication portion of the bilingual collection.

Source pages:

- Week 5: <https://distinct-capricorn-c04.notion.site/Week-5-Layout-Algebra-30488315b6b48073acd8f4a3a89b3b39>
- Week 6: <https://distinct-capricorn-c04.notion.site/Week-6-CuTe-GEMM-30a88315b6b4804b8b8fcfcbdc36554f>
- Week 7: <https://distinct-capricorn-c04.notion.site/Week-7-Communication-32088315b6b4808983bdf39dd7ca922a>

- [ ] **Step 1: Read Weeks 5-7 one page at a time**

  Capture every rendered technical section, formula, layout example, collective
  communication concept, and stable reference. Keep notation definitions with
  the section that first uses them.

- [ ] **Step 2: Write three Chinese-English file pairs**

  Apply the same metadata and cross-link structure as Task 1. Keep CuTe names,
  layout notation, CUDA identifiers, and collective names in their original
  technical spelling while explaining them in Chinese in the primary files.

- [ ] **Step 3: Validate and commit Weeks 5-7**

  Run:

  ```powershell
  $zh = Get-ChildItem docs/course/notes -File | Where-Object Name -Match '^0[5-7]-.*\.md$'
  $en = Get-ChildItem docs/course/source/notes -File | Where-Object Name -Match '^0[5-7]-.*\.en\.md$'
  "zh=$($zh.Count) en=$($en.Count)"
  rg -n -P '\p{Han}' docs/course/source/notes -g '0[5-7]-*.en.md'
  rg -n 'file\.notion\.so|notiontopdf|pvs=|signature=' docs/course/notes docs/course/source/notes
  git diff --check
  git add -- docs/course/notes/0[5-7]-*.md docs/course/source/notes/0[5-7]-*.en.md
  git commit -m "Add HPC layout and communication notes"
  ```

  Expected before commit: `zh=3 en=3`; no Chinese prose or forbidden URLs in the
  English files; no whitespace errors.

### Task 3: Weeks 8-10 Parallelism and Inference

**Files:**
- Create: `docs/course/notes/08-data-expert-parallelism.md`
- Create: `docs/course/notes/09-sequence-parallelism.md`
- Create: `docs/course/notes/10-inference-systems.md`
- Create: `docs/course/source/notes/08-data-expert-parallelism.en.md`
- Create: `docs/course/source/notes/09-sequence-parallelism.en.md`
- Create: `docs/course/source/notes/10-inference-systems.en.md`

**Interfaces:**
- Consumes: the three public pages below and terminology established in Weeks 1-7.
- Produces: the distributed-parallel and inference-system portion of the collection.

Source pages:

- Week 8: <https://distinct-capricorn-c04.notion.site/Week-8-Data-Expert-Parallelism-32688315b6b480c6b66bf9830dfc3cc6>
- Week 9: <https://distinct-capricorn-c04.notion.site/Week-9-Sequence-Parallelism-32d88315b6b480fc8718f6c1b1e5a6fa>
- Week 10: <https://distinct-capricorn-c04.notion.site/Week-10-Inference-Systems-33488315b6b48083b81ecea51648dd7c>

- [ ] **Step 1: Read Weeks 8-10 separately**

  Capture the source's parallelism definitions, communication/data-flow
  descriptions, inference architecture, batching/scheduling concepts, memory
  management, and stable references. Preserve distinctions between training and
  inference semantics.

- [ ] **Step 2: Write three Chinese-English file pairs**

  Use diagrams only when the source relationship would otherwise be ambiguous;
  prefer small Markdown tables and compact equations. Do not claim that the
  repository implements systems described by these notes.

- [ ] **Step 3: Validate and commit Weeks 8-10**

  Run:

  ```powershell
  $zh = Get-ChildItem docs/course/notes -File | Where-Object Name -Match '^(08|09|10)-.*\.md$'
  $en = Get-ChildItem docs/course/source/notes -File | Where-Object Name -Match '^(08|09|10)-.*\.en\.md$'
  "zh=$($zh.Count) en=$($en.Count)"
  rg -n -P '\p{Han}' docs/course/source/notes -g '08-*.en.md' -g '09-*.en.md' -g '10-*.en.md'
  rg -n 'file\.notion\.so|notiontopdf|pvs=|signature=' docs/course/notes docs/course/source/notes
  git diff --check
  git add -- docs/course/notes/08-*.md docs/course/notes/09-*.md docs/course/notes/10-*.md docs/course/source/notes/08-*.en.md docs/course/source/notes/09-*.en.md docs/course/source/notes/10-*.en.md
  git commit -m "Add HPC parallelism and inference notes"
  ```

  Expected before commit: `zh=3 en=3`; no Chinese prose or forbidden URLs in the
  English files; no whitespace errors.

### Task 4: Week 11 and General AI Infra Guide

**Files:**
- Create: `docs/course/notes/11-career-building.md`
- Create: `docs/course/source/notes/11-career-building.en.md`
- Modify: `docs/infra-interview-guide.md`

**Interfaces:**
- Consumes: Week 11 <https://distinct-capricorn-c04.notion.site/Week-11-Career-Building-34388315b6b4808fa70de9b0502a2d15> and reusable technical content already present in the guide.
- Produces: the course-order career note, its English counterpart, and a project-neutral AI Infra learning/interview guide.

- [ ] **Step 1: Read the complete Week 11 source page**

  Capture all sections covering career direction, role expectations, skill
  building, evidence, resume or profile construction, interview practice,
  behavioral communication, and interviewer questions. Record stable links.

- [ ] **Step 2: Write the Chinese and English Week 11 notes**

  Preserve the complete source topic coverage in course order. Use the same
  metadata, cross-link, source-link, and forbidden-URL rules as earlier weeks.

- [ ] **Step 3: Reframe the existing guide**

  Rewrite the opening and remove the current project pitch section, candidate
  positioning, personal contribution rules, repository performance claims, and
  directions to present this repository as interview experience. Retain and
  generalize the reusable technical sections on Softmax, attention, MLA, MoE,
  communication, memory, distributed training, C++, hand-written exercises,
  and performance-evidence methodology.

- [ ] **Step 4: Add the complete interview-oriented Week 11 material**

  Add guide sections for role selection, AI Infra skill map, evidence-building,
  resume principles, preparation loop, behavioral answers, and interviewer
  questions. Link to `course/notes/11-career-building.md` and state that the
  course note contains the same source topics in source order.

- [ ] **Step 5: Validate project-neutral framing and commit**

  Run:

  ```powershell
  rg -n '项目口径|30 秒版本|45 秒版本|一分钟版本|候选人定位|个人贡献|我实现了|本项目|本仓库' docs/infra-interview-guide.md
  rg -n 'course/notes/11-career-building\.md' docs/infra-interview-guide.md
  rg -n -P '\p{Han}' docs/course/source/notes/11-career-building.en.md
  rg -n 'file\.notion\.so|notiontopdf|pvs=|signature=' docs/course/notes/11-career-building.md docs/course/source/notes/11-career-building.en.md
  git diff --check
  ```

  Expected: the project-framing and Chinese-in-English searches have no matches;
  the guide contains one Week 11 link; no forbidden URL or whitespace error is
  present.

  Commit:

  ```powershell
  git add -- docs/course/notes/11-career-building.md docs/course/source/notes/11-career-building.en.md docs/infra-interview-guide.md
  git commit -m "Add career notes and generalize infra guide"
  ```

### Task 5: Bilingual Collection Indexes and Documentation Entry

**Files:**
- Create: `docs/course/hpc-for-ai.md`
- Create: `docs/course/source/hpc-for-ai.en.md`
- Modify: `docs/index.md`

**Interfaces:**
- Consumes: all eleven Chinese-English weekly pairs and the existing syllabus.
- Produces: complete bilingual navigation and a project-neutral main docs entry.

- [ ] **Step 1: Create the Chinese collection index**

  Include parent source metadata, snapshot date, the existing syllabus link,
  English-index link, and an ordered eleven-row table. Each row links one Chinese
  weekly file and summarizes its learning role in one sentence.

- [ ] **Step 2: Create the English collection index**

  Mirror the eleven-row navigation in English, link the English syllabus source
  and Chinese collection index, and keep descriptions free of Chinese prose.

- [ ] **Step 3: Update the main docs index**

  Link `course/hpc-for-ai.md` from `docs/index.md`. Describe the repository as an
  AI Infra knowledge collection, and replace the current statement about
  repository facts and personal contribution evidence with general study and
  interview-practice wording.

- [ ] **Step 4: Validate complete file coverage and navigation**

  Run:

  ```powershell
  $zh = Get-ChildItem docs/course/notes -File -Filter '*.md'
  $en = Get-ChildItem docs/course/source/notes -File -Filter '*.en.md'
  "zh=$($zh.Count) en=$($en.Count)"
  rg -n '^\| (1[01]|[1-9]) ' docs/course/hpc-for-ai.md docs/course/source/hpc-for-ai.en.md
  rg -n 'info-7375-syllabus' docs/course/hpc-for-ai.md docs/course/source/hpc-for-ai.en.md
  rg -n -P '\p{Han}' docs/course/source/hpc-for-ai.en.md docs/course/source/notes
  rg -n 'file\.notion\.so|notiontopdf|pvs=|signature=' docs/course
  rg -n 'course/hpc-for-ai\.md' docs/index.md
  git diff --check
  ```

  Expected: `zh=11 en=11`; both indexes contain eleven numbered rows and a
  syllabus link; English files contain no Chinese prose; forbidden URL scan has
  no matches; the main index contains one collection link; no whitespace errors.

- [ ] **Step 5: Commit the collection indexes**

  ```powershell
  git add -- docs/course/hpc-for-ai.md docs/course/source/hpc-for-ai.en.md docs/index.md
  git commit -m "Index the HPC for AI note collection"
  ```

### Task 6: Final Repository Verification

**Files:**
- Verify: `docs/course/`
- Verify: `docs/infra-interview-guide.md`
- Verify: `docs/index.md`

**Interfaces:**
- Consumes: every committed documentation task.
- Produces: final evidence that the collection is complete, linked, project-neutral, and does not regress the repository test suite.

- [ ] **Step 1: Verify the committed documentation tree**

  Run:

  ```powershell
  git status --short
  rg --files docs/course/notes docs/course/source/notes
  rg -n '项目口径|候选人定位|个人贡献|我实现了|本项目|本仓库' docs/infra-interview-guide.md
  rg -n 'file\.notion\.so|notiontopdf|pvs=|signature=' docs/course
  git diff --check HEAD~5..HEAD
  ```

  Expected: clean working tree; 22 weekly files; project-framing and forbidden
  URL searches have no matches; no whitespace errors across the five
  implementation commits.

- [ ] **Step 2: Run the full repository test suite**

  Run:

  ```powershell
  .\.venv\Scripts\python.exe -m pytest -ra --strict-markers -W error::UserWarning
  ```

  Expected: exit code 0; hardware-dependent tests may remain explicitly skipped.

- [ ] **Step 3: Confirm local and upstream state**

  Run:

  ```powershell
  git branch --show-current
  git rev-list --left-right --count 'origin/main...HEAD'
  git status --short
  ```

  Expected before any push: branch `main`, no local modifications, and only the
  new documentation commits ahead of `origin/main`.
