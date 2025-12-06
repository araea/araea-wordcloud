use araea_wordcloud::{ColorScheme, WordCloudBuilder, WordInput};
use rand::Rng;
use std::fs;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut words = generate_dense_chinese_data();
    println!("Generated {} Chinese words.", words.len());

    let start = Instant::now();

    let width = 1200;
    let height = 800;

    println!("Building word cloud ({}x{})...", width, height);

    let count_len = words.len();
    let dynamic_max_font_size = if count_len < 20 {
        150.0
    } else if count_len < 50 {
        120.0
    } else {
        100.0
    };

    let max_weight = words.first().map(|w| w.weight).unwrap_or(100.0);
    for word in &mut words {
        if max_weight > 100.0 {
            word.weight = word.weight.sqrt() * 10.0;
        }
    }

    let scheme = ColorScheme::Default;
    let wordcloud = WordCloudBuilder::new()
        .size(width, height)
        .color_scheme(scheme)
        .background(scheme.background_color())
        .padding(2)
        .angles(vec![0.0, 90.0])
        .word_spacing(2.0)
        .font_size_range(14.0, dynamic_max_font_size)
        // .seed(2025)
        .build(&words)?;

    let output_path = "output_chinese_dense.png";
    fs::write(output_path, wordcloud.to_png(2.0)?)?;

    println!("Success! Time elapsed: {:?}", start.elapsed());
    println!("Saved to {}", output_path);

    Ok(())
}

fn generate_dense_chinese_data() -> Vec<WordInput> {
    let mut words = Vec::new();
    let mut rng = rand::rng();

    // 核心超大词，权重分布更均匀，突出重点词汇，同时避免权重集中
    let core_keywords = vec![
        "Rust编程",
        "高性能",
        "内存安全",
        "WebAssembly",
        "系统级",
        "无GC",
        "所有权",
        "生命周期",
        "借用检查",
        "并发模型",
    ];
    for w in &core_keywords {
        // 在85~100之间随机，增加大词的字体差异感
        words.push(WordInput::new(*w, rng.random_range(85.0..=100.0)));
    }

    // 重要相关概念，权重调整为55~80，增强视觉层次感
    let concepts = vec![
        "Cargo",
        "Crates.io",
        "Tokio",
        "Actix",
        "Serde",
        "Diesel",
        "Async/Await",
        "Trait",
        "Struct",
        "Enum",
        "Pattern Matching",
        "Zero-cost",
        "安全性",
        "跨平台",
        "嵌入式",
        "网络服务",
        "命令行工具",
        "错误处理",
        "Result",
        "Option",
    ];
    for w in &concepts {
        words.push(WordInput::new(*w, rng.random_range(55.0..=80.0)));
    }

    // 常用基础词汇，分三轮加入，权重均匀分布40~55，避免视觉“拥挤”
    let common = vec![
        "编译",
        "运行",
        "测试",
        "文档",
        "模块",
        "函数",
        "闭包",
        "迭代器",
        "集合",
        "字符串",
        "指针",
        "引用",
        "智能指针",
        "Box",
        "Rc",
        "Arc",
        "Mutex",
        "Channel",
        "Future",
        "Stream",
        "宏定义",
        "属性",
        "泛型",
        "Community",
        "Foundation",
        "Docs",
        "Book",
        "Examples",
        "Tutorial",
    ];
    for _ in 0..3 {
        for w in &common {
            words.push(WordInput::new(*w, rng.random_range(40.0..=55.0)));
        }
    }

    // 填充词汇，丰富词云细节，权重范围调整为20~35，减少过密感
    let fillers = vec![
        "代码", "逻辑", "数据", "接口", "实现", "调用", "返回", "参数", "类型", "变量", "常量",
        "静态", "动态", "链接", "库", "包", "依赖", "版本", "发布", "构建", "优化", "调试", "日志",
        "监控", "设计", "架构", "模式", "算法", "结构", "效率", "速度", "稳定", "扩展", "维护",
        "重构", "迁移", "学习", "曲线", "入门", "精通",
    ];
    for _ in 0..8 {
        // 略减少次数，控制填充词数量避免拥挤
        for w in &fillers {
            words.push(WordInput::new(*w, rng.random_range(20.0..=35.0)));
        }
    }

    words
}
