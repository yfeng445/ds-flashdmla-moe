# AI Infra 文档

`docs/` 按使用场景分成课程、面试和论文三部分。课程资料保留完整学习路径；面试资料按一份题库一个 Markdown 文件组织；论文目录用于保存可合法分发的原文、翻译及阅读索引。

| 目录 | 内容 | 当前入口 |
| --- | --- | --- |
| [`courses/`](courses/NEU_INFO_7375/index.md) | 课程大纲、周笔记、专题讲义、练习和源文档 | [NEU INFO 7375](courses/NEU_INFO_7375/index.md) |
| [`interviews/`](interviews/index.md) | AI Infra 知识路线与逐份 Q&A 题库 | [面试资料入口](interviews/index.md) |
| [`papers/`](papers/index.md) | Infra 论文原文、翻译件与可追溯参考资料 | [论文与参考资料](papers/index.md) |

## 目录约定

```text
docs/
├── courses/
│   └── NEU_INFO_7375/
│       ├── index.md
│       ├── syllabus.md
│       ├── chapters/
│       ├── notes/
│       ├── source/
│       └── exercises.md
├── interviews/
│   ├── index.md
│   └── *-qa.md
└── papers/
    └── index.md
```

- 新课程沿用 `NEU_INFO_7375` 的骨架；不要求每门课都有相同数量的章节或笔记。
- 面试题库按主题拆成独立的 `*-qa.md`，方便复习、检索和后续从其他资料中增量迁入问题。
- 论文原文或翻译件进入 `papers/` 前，应保留来源、版本和授权信息；无法确认再分发权限时只记录外部链接与阅读笔记。
- 实施计划和设计记录位于仓库根目录的 `superpowers/`，不属于读者文档。
