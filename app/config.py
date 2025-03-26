import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class ScrapingConfig:
    # AWS Configuration
    aws_region: str = os.getenv('AWS_REGION', 'us-east-2')
    s3_bucket: str = os.getenv('S3_BUCKET')
    
    # Scraping Configuration
    base_url: str = "https://www.banxico.org.mx/publicaciones-y-prensa/informes-trimestrales/informes-trimestrales-precios.html"
    download_path: str = "/tmp/pdfs"
    max_retries: int = 3
    timeout: int = 30
    
    # Processing Configuration
    max_concurrent_downloads: int = 5
    chunk_size: int = 8192  # 8KB chunks for download
    
    # Monitoring
    enable_logging: bool = True
    log_level: str = "INFO"

@dataclass
class ProcessingConfig:
    # Database Configuration
    db_host: Optional[str] = os.getenv('DB_HOST')
    db_port: int = int(os.getenv('DB_PORT', '5432'))
    db_name: str = os.getenv('DB_NAME', 'clasificador')
    db_user: str = os.getenv('DB_USER', 'admin')
    db_password: Optional[str] = os.getenv('DB_PASSWORD')
    
    # Processing Settings
    batch_size: int = 10
    max_workers: int = 4
    
    # Output Configuration
    output_format: str = "json"
    structured_data_path: str = "/tmp/processed"

# Instancias de configuración
scraping_config = ScrapingConfig()
processing_config = ProcessingConfig()

# Logging Configuration
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': {
        'default': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
        },
    },
    'loggers': {
        '': {
            'handlers': ['default'],
            'level': 'INFO',
            'propagate': True
        }
    }
} 