use araea_wordcloud::{ColorScheme, MaskShape, WordCloudBuilder, WordInput};
use std::fs;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let output_path = "output_mask_heart.png";

    let width = 800;
    let height = 800;

    println!("Generating words...");
    let mut words = Vec::new();
    for i in 0..300 {
        words.push(WordInput::new("Love", 100.0));
        words.push(WordInput::new("Rust", 80.0));
        words.push(WordInput::new("Heart", 60.0 + (i as f32 % 40.0)));
        words.push(WordInput::new("Mask", 40.0));
    }

    println!("Building word cloud...");

    let scheme = ColorScheme::Default;
    let wordcloud = WordCloudBuilder::new()
        .size(width, height)
        .mask_preset(MaskShape::Heart)
        .color_scheme(scheme)
        .background(scheme.background_color())
        .padding(2)
        .font_size_range(12.0, 80.0)
        .build(&words)?;

    println!("Saving to {}...", output_path);
    fs::write(output_path, wordcloud.to_png(2.0)?)?;

    println!("Done! Check {}", output_path);
    Ok(())
}
