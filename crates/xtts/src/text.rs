//! Text front-end: number/abbreviation normalization, XTTS' multilingual
//! cleaners, sentence splitting and the BPE tokenizer.
//!
//! Coqui expands numbers with `num2words`; here French and English go
//! through the `tn` rules (numbers, times, dates, amounts, units,
//! Markdown), and the XTTS abbreviation and symbol tables apply to every
//! Latin-script language. Chinese, Japanese and Korean need a
//! transliteration (pinyin, romaji, hangul romanization) this crate does
//! not do, so they are rejected.

use crate::text_tables::{ABBREVIATIONS, CHAR_LIMITS, SYMBOLS};
use anyhow::{Result, anyhow, bail};
use regex::Regex;
use std::collections::HashMap;
use std::sync::LazyLock;

/// Languages of XTTS-v2 this crate accepts.
pub const LANGUAGES: &[&str] = &[
    "en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "hu", "hi",
];

type Rules = HashMap<&'static str, Vec<(Regex, &'static str)>>;

fn compile(table: &'static [(&'static str, &'static [(&'static str, &'static str)])]) -> Rules {
    table
        .iter()
        .map(|(lang, pairs)| {
            let rules = pairs
                .iter()
                .map(|(rx, rep)| (Regex::new(rx).expect("table regex"), *rep))
                .collect();
            (*lang, rules)
        })
        .collect()
}

static ABBREV: LazyLock<Rules> = LazyLock::new(|| compile(ABBREVIATIONS));
static SYMBOL: LazyLock<Rules> = LazyLock::new(|| compile(SYMBOLS));
static SPACES: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\s+").unwrap());

/// `fr-FR` → `fr`; `zh-cn` → `zh`.
pub fn base_lang(lang: &str) -> String {
    lang.split(['-', '_']).next().unwrap_or("").to_lowercase()
}

pub fn char_limit(lang: &str) -> usize {
    CHAR_LIMITS
        .iter()
        .find(|(l, _)| *l == lang)
        .map(|(_, n)| *n)
        .unwrap_or(250)
}

/// Spells out numbers, symbols and Markdown with `tn` (French, English).
/// Other languages only get `tn`'s Markdown and emoji handling.
pub fn normalize(text: &str, lang: &str) -> String {
    tn::normalize_text(text, tn::Lang::from_code(lang), tn::Mode::Strict, None)
}

/// XTTS' `multilingual_cleaners` minus `num2words`: quotes removed,
/// lowercase, abbreviations, symbols, collapsed whitespace.
pub fn clean(text: &str, lang: &str) -> String {
    let mut t = text.replace('"', "");
    if lang == "tr" {
        t = t.replace('İ', "i").replace('Ö', "ö").replace('Ü', "ü");
    }
    t = t.to_lowercase();
    if let Some(rules) = ABBREV.get(lang) {
        for (rx, rep) in rules {
            t = rx.replace_all(&t, *rep).into_owned();
        }
    }
    if let Some(rules) = SYMBOL.get(lang) {
        for (rx, rep) in rules {
            t = rx.replace_all(&t, *rep).into_owned();
            t = t.replace("  ", " ");
        }
        t = t.trim().to_string();
    }
    SPACES.replace_all(&t, " ").trim().to_string()
}

/// A chunk without its final period. XTTS often reads a final `.` out
/// loud ("point", "punto") or babbles a syllable after it; without it the
/// stop code comes right after the last word (Parakeet WER on
/// `scripts/wer.py`: fr 1.1 → 0.6 %, en 0.5 → 0.0 %). `?` and `!` stay, they
/// carry the intonation; so does an ellipsis.
pub fn drop_final_period(chunk: &str) -> String {
    let t = chunk.trim_end();
    match t.strip_suffix('.') {
        Some(rest) if !rest.ends_with('.') => rest.trim_end().to_string(),
        _ => t.to_string(),
    }
}

/// Splits text into chunks of at most `limit` characters, at sentence ends
/// when possible (Coqui's `split_sentence`, with a punctuation splitter in
/// place of spaCy): sentences are packed together while they fit, longer
/// ones are wrapped at spaces.
pub fn split_sentences(text: &str, limit: usize) -> Vec<String> {
    let text = text.trim();
    if text.chars().count() < limit {
        return if text.is_empty() {
            vec![]
        } else {
            vec![text.to_string()]
        };
    }
    let mut sentences = Vec::new();
    let mut cur = String::new();
    let chars: Vec<char> = text.chars().collect();
    for (i, &c) in chars.iter().enumerate() {
        cur.push(c);
        let end = matches!(c, '.' | '!' | '?' | '…' | ';' | '。' | '！' | '？' | '\n');
        let next_space = chars.get(i + 1).is_none_or(|n| n.is_whitespace());
        if end && next_space {
            let s = cur.trim().to_string();
            if !s.is_empty() {
                sentences.push(s);
            }
            cur.clear();
        }
    }
    if !cur.trim().is_empty() {
        sentences.push(cur.trim().to_string());
    }

    let mut out: Vec<String> = Vec::new();
    let len = |s: &str| s.chars().count();
    for s in sentences {
        if len(&s) > limit {
            out.extend(wrap(&s, limit));
        } else if let Some(last) = out.last_mut().filter(|l| len(l) + 1 + len(&s) <= limit) {
            last.push(' ');
            last.push_str(&s);
        } else {
            out.push(s);
        }
    }
    out
}

fn wrap(s: &str, width: usize) -> Vec<String> {
    let mut lines = Vec::new();
    let mut cur = String::new();
    for word in s.split_whitespace() {
        let n = cur.chars().count();
        if n > 0 && n + 1 + word.chars().count() > width {
            lines.push(std::mem::take(&mut cur));
        }
        if !cur.is_empty() {
            cur.push(' ');
        }
        cur.push_str(word);
    }
    if !cur.is_empty() {
        lines.push(cur);
    }
    lines
}

pub struct TextTokenizer {
    tok: tokenizers::Tokenizer,
    pub start: u32,
    pub stop: u32,
}

impl TextTokenizer {
    pub fn from_json(json: &str) -> Result<Self> {
        let tok = tokenizers::Tokenizer::from_bytes(json.as_bytes()).map_err(|e| anyhow!("{e}"))?;
        let id = |t: &str| tok.token_to_id(t).ok_or_else(|| anyhow!("no {t} token"));
        Ok(Self {
            start: id("[START]")?,
            stop: id("[STOP]")?,
            tok,
        })
    }

    /// Token ids of one chunk of already normalized text (XTTS
    /// `VoiceBpeTokenizer.encode`, after `.strip().lower()`), without the
    /// start/stop tokens.
    pub fn encode(&self, text: &str, lang: &str) -> Result<Vec<u32>> {
        let lang = base_lang(lang);
        if !LANGUAGES.contains(&lang.as_str()) {
            bail!(
                "language '{lang}' is not supported (supported: {})",
                LANGUAGES.join(", ")
            );
        }
        let text = clean(text.trim(), &lang).to_lowercase();
        let tagged = format!("[{lang}]{text}").replace(' ', "[SPACE]");
        let enc = self.tok.encode(tagged, true).map_err(|e| anyhow!("{e}"))?;
        Ok(enc.get_ids().to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// covers: REQ-TXT-001
    #[test]
    fn tables_compile() {
        assert!(ABBREV.len() >= 15);
        assert!(SYMBOL.len() >= 15);
    }

    /// covers: REQ-TXT-001
    #[test]
    fn cleaners() {
        assert_eq!(
            clean("Mme. Dupont & \"Fils\"  à 50%", "fr"),
            "madame dupont et fils à 50 pour cent"
        );
        assert_eq!(clean("Dr. Smith", "en"), "doctor smith");
    }

    /// covers: REQ-TXT-002
    #[test]
    fn numbers_are_spelled_out() {
        let fr = normalize("Il est 14h30, ça coûte 12,50 €.", "fr");
        assert!(fr.contains("quatorze heures trente"), "{fr}");
        assert!(fr.contains("douze euros cinquante"), "{fr}");
        let en = normalize("It costs $3.", "en");
        assert!(en.contains("three dollars"), "{en}");
        assert_eq!(normalize("Bonjour à tous.", "fr"), "Bonjour à tous.");
    }

    /// covers: REQ-TXT-004
    #[test]
    fn final_period() {
        assert_eq!(drop_final_period("Bonjour à tous. "), "Bonjour à tous");
        assert_eq!(drop_final_period("Vraiment ?"), "Vraiment ?");
        assert_eq!(drop_final_period("Et alors..."), "Et alors...");
        assert_eq!(drop_final_period("Un. Deux."), "Un. Deux");
    }

    /// covers: REQ-TXT-003
    #[test]
    fn splitting() {
        let t = "Une phrase. Une autre phrase! Et une troisième?";
        assert_eq!(
            split_sentences(t, 30),
            vec!["Une phrase. Une autre phrase!", "Et une troisième?"]
        );
        assert_eq!(split_sentences("court", 30), vec!["court"]);
        let long = "mot ".repeat(20);
        assert!(
            split_sentences(&long, 30)
                .iter()
                .all(|s| s.chars().count() <= 30)
        );
    }
}
