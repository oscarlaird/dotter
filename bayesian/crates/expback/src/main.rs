use mistralrs::{IsqBits, ModelBuilder, TextModelBuilder, TextMessages, TextMessageRole, Model};

#[tokio::main]
async fn main() -> mistralrs::error::Result<()> {
    let model = ModelBuilder::new("Qwen/Qwen3-4B")
        .with_auto_isq(IsqBits::Four)
        .build()
        .await?;
    let response = model.chat("What is Rust's ownership model?").await?;
    println!("{response}");
    Ok(())
}
