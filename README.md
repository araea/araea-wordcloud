# araea-wordcloud

Rust 词云库，可将带权词语排布为 SVG 或 PNG。布局使用像素掩码和阿基米德螺线；无法放入画布的词会被跳过。

## 安装

```toml
[dependencies]
araea-wordcloud = "0.1"
```

## 示例

```rust
use araea_wordcloud::generate;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let words = [("Rust", 100.0), ("Code", 60.0)];
    let cloud = generate(&words)?;
    std::fs::write("output.svg", cloud.to_svg())?;
    std::fs::write("output.png", cloud.to_png(2.0)?)?;
    Ok(())
}
```

空词和非正权重会被忽略；过滤后没有词时返回错误。

## 布局与输出

复杂布局可使用 `WordCloudBuilder` 设置画布尺寸、字体、字号范围、掩码、颜色、旋转角度、间距和随机种子。支持内置 Circle、Cloud、Heart、Skull、Star、Triangle 掩码，也可加载 SVG、PNG 或 JPEG 掩码。竖排 CJK 可通过 `vertical_writing(true)` 启用。

默认画布为 800×600，字号范围为 10–100，间距为 5 px，字体为内置 HarmonyOS Sans SC Bold。`trim(true)` 按词语包围盒裁切成图；`trim_margin(px)` 设置裁切边距。SVG 与 PNG 使用相同裁切边界。

`to_svg()` 返回 SVG 字符串，`to_png(scale)` 返回 PNG 字节。`WordCloud` 提供画布尺寸、视口、背景和每个词的位置、字号、颜色、旋转角度及包围盒；`content_bounds()` 返回内容边界。

## 测试

```sh
cargo test
cargo run --example simple
cargo run --example trim
cargo run --example chinese_vertical
```

## 许可证

可按 [Apache-2.0](LICENSE-APACHE) 或 [MIT](LICENSE-MIT) 使用。
