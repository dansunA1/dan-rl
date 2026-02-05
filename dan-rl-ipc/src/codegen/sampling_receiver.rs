#![feature(generic_const_exprs)]
//! Binary receiver for sampling messages
//!
//! This binary connects to a sampling service, counts received messages,
//! and reports the total count when terminated (SIGTERM).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;
use signal_hook::consts::{SIGINT, SIGTERM};
use signal_hook::iterator::Signals;
use clap::{Parser, ArgAction};
use tracing::{debug, Level};
use tracing_appender::non_blocking;
use std::sync::Mutex;
use tracing_appender::non_blocking::WorkerGuard;

use crate::server::Server;
use crate::prelude::{PayloadScheme};

static TRACING_GUARD: once_cell::sync::OnceCell<Mutex<Option<WorkerGuard>>> =
    once_cell::sync::OnceCell::new();

fn init_tracing() {
    let (writer, guard) = non_blocking(std::io::stdout());

    // Keep guard alive forever
    TRACING_GUARD.set(Mutex::new(Some(guard))).unwrap();

    let max_level = Level::DEBUG;

    tracing_subscriber::fmt()
        .with_writer(writer)
        .with_max_level(max_level)
        .with_target(false)
        .with_thread_ids(false)
        .with_thread_names(false)
        .init();
}

#[derive(Parser)]
#[command(
    author = "Dan Sun <dansun@alfa1.io>",
    version = "0.1.0",
    about = "A receiver that counts sampling messages and reports total count"
)]
struct Cli {
    #[arg(long, default_value = "hftrlserver")]
    sampling_channel: String,

    #[arg(long, default_value_t = 10)]
    timeout_secs: u64,

    #[arg(long, action = ArgAction::SetTrue)]
    verbose: bool,
}

fn entry<S:PayloadScheme>() -> Result<(), Box<dyn std::error::Error>> 
where
    [();S::SDIM]: Sized,
    [();S::ADIM]: Sized,
    [();S::RDIM]: Sized,
    [();S::N_INFO]: Sized,
    [();S::N]: Sized,
{
    let cli = Cli::parse();
    if cli.verbose {
        init_tracing();
    }
    
    let stop = Arc::new(AtomicBool::new(false));
    let stop_clone = stop.clone();
    
    // Handle termination signals
    let mut signals = Signals::new(&[SIGTERM, SIGINT])?;
    std::thread::spawn(move || {
        for _ in signals.forever() {
            stop_clone.store(true, Ordering::Relaxed);
            break;
        }
    });
    
    let server = Server::new()?;
    let receiver = server.sampling_receiver::<S>(cli.sampling_channel.clone())?;
    
    eprintln!("Receiver initialized, waiting for first message...");
    
    // Wait for first message to ensure connection is established
    let mut connected = false;
    let timeout = Duration::from_secs(cli.timeout_secs);
    let start = std::time::Instant::now();
    
    while !connected && !stop.load(Ordering::Relaxed) && start.elapsed() < timeout {
        if receiver.receive(|_| {}).unwrap().is_some() {
            connected = true;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    
    if !connected {
        eprintln!("ERROR: Failed to receive first message within {} seconds", cli.timeout_secs);
        std::process::exit(1);
    }
    
    // Count messages
    
    while !stop.load(Ordering::Relaxed) {
        if receiver.receive(|_| {}).unwrap().is_some() {
            break;
        } else {
            std::thread::sleep(Duration::from_millis(1));
        }
    }
    let mut count = 1u64;
    let start = std::time::Instant::now();
    while !stop.load(Ordering::Relaxed) {
        if let Some(_payload) = receiver.receive(|p| {
            if cli.verbose {
                debug!(?p);
            }
        }).unwrap() {
            count += 1;
        } else {
            std::thread::sleep(Duration::from_millis(1));
        }
    }
    let duration = start.elapsed();
    if let Some(mutex) = TRACING_GUARD.get() {
        // Drop the guard by taking it
        let _ = mutex.lock().unwrap().take();
    }
    // Report total count
    println!("Total samples received: {}", count);
    println!("Time taken: {:?}", duration);
    println!("Rate: {} samples/s", count as f64 / duration.as_secs() as f64);
    Ok(())
}

