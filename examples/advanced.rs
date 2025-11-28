use araea_wordcloud::{WordCloudBuilder, WordInput};
use std::fs;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let words = vec![
        WordInput::new("Custom", 90.0),
        WordInput::new("Colors", 80.0),
        WordInput::new("Seed", 70.0),
        WordInput::new("Fixed", 60.0),
        WordInput::new("Layout", 50.0),
    ];

    let wordcloud = WordCloudBuilder::new()
        .size(600, 400)
        .background("#1a1a1a")
        .colors(vec!["#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#00FFFF"])
        .seed(42)
        .angles(vec![0.0])
        .font_size_range(20.0, 100.0)
        .build(&words)?;

    fs::write("output_advanced.png", wordcloud.to_png(1.0)?)?;
    println!("Generated advanced word cloud: output_advanced.png");

    Ok(())
}
