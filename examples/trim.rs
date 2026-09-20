use araea_wordcloud::{WordCloudBuilder, WordInput};
use std::fs;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let words = vec![
        WordInput::new("Rust", 100.0),
        WordInput::new("Code", 60.0),
        WordInput::new("Fast", 40.0),
        WordInput::new("Safe", 30.0),
        WordInput::new("Cloud", 20.0),
    ];

    // 画布只决定词排得开不开；`trim` 让成图跟着内容走，不留下大片背景。
    let wordcloud = WordCloudBuilder::new()
        .size(800, 600)
        .trim(true)
        .trim_margin(16)
        .seed(42)
        .build(&words)?;

    fs::write("output_trim.png", wordcloud.to_png(2.0)?)?;
    println!(
        "canvas {}x{} -> output {}x{}",
        wordcloud.width, wordcloud.height, wordcloud.viewport.width, wordcloud.viewport.height
    );

    Ok(())
}
