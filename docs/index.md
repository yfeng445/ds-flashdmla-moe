# AI Infra 文档

`docs/` 是本仓库的技术资料库，由课程与论文两部分组成。课程资料保留完整学习路径；论文目录保存本地参考副本、翻译件、来源清单和阅读索引。

本目录只维护 AI Infra 技术内容。面试准备、简历、岗位分析和个人项目表述不属于本仓库范围。

| 目录 | 内容 | 当前入口 |
| --- | --- | --- |
| [`courses/`](courses/NEU_INFO_7375/index.md) | 课程大纲、周笔记、专题讲义、练习和源文档 | [NEU INFO 7375](courses/NEU_INFO_7375/index.md) |
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
└── papers/
    ├── index.md
    ├── catalog.md
    ├── manifest.yaml
    ├── attention-kernels/
    ├── mla-transformers/
    ├── moe/
    ├── distributed-training/
    ├── serving/
    ├── scaling-foundations/
    └── books/
```

- 新课程沿用 `NEU_INFO_7375` 的骨架；不要求每门课都有相同数量的章节或笔记。
- 论文原文或翻译件进入 `papers/` 时，应保留来源、版本、翻译状态和权利归属。外部资料不因进入本仓库而改用仓库的 MIT License；公开再分发前仍需逐份核对许可。
- 实施计划和设计记录位于仓库根目录的 `superpowers/`，不属于读者文档。
