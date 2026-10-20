use std::error::Error;
use tuneout_resampler::{FilterLength, FilterLengthPolicy};

pub fn filter_length(value: Option<&String>) -> Result<FilterLength, Box<dyn Error>> {
    Ok(match value.map(String::as_str).unwrap_or("standard") {
        "standard" => FilterLength::Standard,
        "long" => FilterLength::Long,
        "extra-long" => FilterLength::ExtraLong,
        taps => FilterLength::Custom(taps.parse()?),
    })
}

pub fn filter_length_policy(value: Option<&String>) -> Result<FilterLengthPolicy, Box<dyn Error>> {
    match value.map(String::as_str).unwrap_or("generic") {
        "generic" => Ok(FilterLengthPolicy::Generic),
        "fixed" => Ok(FilterLengthPolicy::Fixed),
        _ => Err("POLICY must be generic or fixed".into()),
    }
}
