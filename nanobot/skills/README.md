# nanobot Skills

This directory contains built-in skills that extend nanobot's capabilities.

## Skill Format

Each skill is a directory containing a `SKILL.md` file with:
- YAML frontmatter (name, description, metadata)
- Markdown instructions for the agent

## Attribution

These skills are adapted from [OpenClaw](https://github.com/openclaw/openclaw)'s skill system.
The skill format and metadata structure follow OpenClaw's conventions to maintain compatibility.

## Available Skills

| Skill | Description |
|-------|-------------|
| `cron` | Schedule reminders and recurring tasks |
| `github` | Interact with GitHub using the `gh` CLI |
| `memory` | Two-layer memory system with grep-based recall |
| `summarize` | Summarize URLs, files, and YouTube videos |
| `skill-creator` | Create new skills |
