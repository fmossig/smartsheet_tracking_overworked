# Smartsheet Tracker & Reporting System

A Python-based automation system for monitoring and tracking changes in Smartsheet tables for Amazon Cases management. The system automatically detects phase transitions, logs changes, and generates professional PDF reports with comprehensive analytics.

## Features

- **Multi-Sheet Tracking**: Monitors 9 product group sheets (NA, NF, NH, NP, NT, NV, NM, BUNDLE_FAN, BUNDLE_COOLER)
- **Phase Monitoring**: Tracks 5 phases of case processing with user assignments
- **Automated Reports**: Generates weekly and monthly PDF reports with charts and metrics
- **Comprehensive Error Handling**: Graceful degradation with row-level error isolation
- **Executive Summary**: High-level overview with period-over-period comparisons
- **Advanced Analytics**: User productivity, phase progression funnels, trend analysis
- **Special Activities Tracking**: Separate tracking for special activities with category breakdowns
- **GitHub Actions Integration**: Automated scheduled runs with manual trigger options

## Requirements

- Python 3.10+
- Smartsheet API access token
- Dependencies listed in `requirements.txt`

## Installation

1. Clone this repository
2. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Create a `.env` file with your Smartsheet token:
   ```
   SMARTSHEET_TOKEN=your_token_here
   ```

## Usage

### Track Changes
```bash
python smartsheet_tracker.py
```

### Generate Reports
```bash
# Weekly report
python smartsheet_report.py --weekly

# Monthly report
python smartsheet_report.py --monthly

# Custom date range
python smartsheet_report.py --start-date 2024-01-01 --end-date 2024-01-31
```

### Diagnostics
```bash
python smartsheet_diagnostic.py
```

### Reset State
```bash
python reset_state.py
```

## Project Structure

- `smartsheet_tracker.py` - Main tracking script
- `smartsheet_report.py` - Report generation script
- `constants.py` - Configuration constants (Sheet IDs, colors, etc.)
- `error_collector.py` - Error handling and collection framework
- `historical_data_loader.py` - Historical data analysis
- `*_calculator.py` - Various metric calculators
- `tests/` - Unit tests
- `.github/workflows/` - GitHub Actions automation

## GitHub Actions

The system runs automatically via GitHub Actions:
- **Hourly**: Change tracking
- **Weekly**: Monday morning report generation
- **Monthly**: First day of month report generation
- **Manual**: Trigger via workflow dispatch

## Configuration

Edit `constants.py` to configure:
- Sheet IDs
- Phase field definitions
- Color schemes
- Product counts
- Report settings

## Error Handling

The system includes comprehensive error handling:
- API timeouts and rate limiting
- Missing or invalid data
- Sheet access issues
- Unicode and special character handling
- Row-level error isolation (continues processing if one row fails)

## Reports Include

- Executive summary with key metrics
- Group detail pages with phase progression
- User activity analysis with trends
- Special activities breakdown
- Marketplace activity metrics
- Error and warning summaries
- Period-over-period comparisons

## Development

Run tests:
```bash
python -m pytest tests/
```

## License

[Specify your license]

## Authors

Noctua Returns Department
Enhanced by Automaker
