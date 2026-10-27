# language

Reads and normalizes the language of a text, as an ISO 639-1 code. One
implementation for every product that keys content, doctrine or translations
by language.

```
go get github.com/lalternativefabrique/searchkit/language@v0.2.0
```

| Function | Does |
|---|---|
| `Normalize(tag)` | `"en-US"` → `"en"`, `" FR "` → `"fr"`, anything that is not a two-letter code → `""` |
| `Detect(texts...)` | statistical detection ([whatlanggo](https://github.com/abadojack/whatlanggo)), `""` below 40 runes or when the guess is not reliable |
| `Or(language, fallback)` | `fallback` when `language` is `""` |

`Detect` is offline and costs microseconds: run it on the material itself,
not on a title. A three-word title reads as half the languages of Europe.
