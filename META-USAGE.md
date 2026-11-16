# META-USAGE — використання ШІ в проєкті

Згідно з Методичними рекомендаціями (розділ 6) тут декларується весь згенерований код.

| Дата | Інструмент | Що зроблено | Файли |
|---|---|---|---|
| 2026-11-15 | Claude (Anthropic), Claude Opus 5.5 | Відновлення формату архіву розбором еталонних архівів викладача (ключ з паролю в team_config.json, розшифрування, аналіз структури payload) | `docs/FORMAT_SPEC.md` |
| 2026-11-15 | Claude (Anthropic), Claude Opus 5.5 | Виправлення за рев'ю PR #1 | `sarc/*.py`, `tests/*.py` |
| 2026-11-16 | Claude (Anthropic), Claude Opus 5.5 | Виправлення за рев'ю PR #2: ДП за цінністю, тимчасовий файл для блоків, обробка помилок читання/запису, медіана часу | `sarc/packing.py`, `sarc/archive.py`, `sarc/lz77.py`, `tools/measure.py`, `tests/*.py` |
| 2026-11-16 | Claude (Anthropic), Claude Opus 5.5 | Архіватор: кодер Гаффмана, LZ77 (жадібний і лінивий), рюкзак ДП/жадібний, команда archive, тести 12 кейсів C1, скрипт вимірювань | `sarc/huffman.py`, `sarc/codec.py`, `sarc/lz77.py`, `sarc/packing.py`, `sarc/analysis.py`, `sarc/archive.py`, `sarc/cli.py`, `tests/test_acceptance.py`, `tools/measure.py`, `docs/RESULTS.md` |
| 2026-11-15 | Claude (Anthropic), Claude Opus 5.5 | Каркас пакета, розархіватор з кодами відмов, тести на еталонних архівах, CI | `sarc/*.py`, `tests/*.py`, `.github/workflows/ci.yml`, `pyproject.toml`, `README.md` |

Перевірку виконано автоматичними тестами: усі еталонні архіви `data/reference/ref_*.arc`
розпаковуються з правильними CRC32, усі `bad_*.arc` дають очікувані коди з `expected_failures.json`.

Кожен учасник, що використовує ШІ у своїй зоні, додає сюди рядок: дата, інструмент, що саме, які файли.
