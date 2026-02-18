#!/bin/bash

# Run migrations
echo "Running migrations..."
python manage.py migrate

# Import all data
echo "Importing all data..."
IMPORT_COMMANDS=(
    "import_BAC" "import_CHA" "import_CYB" "import_CZA" "import_HJC"
    "import_JOA" "import_KAB" "import_KKA" "import_KYI" "import_KZA"
    "import_OT" "import_OU" "import_OV" "import_OW" "import_OZ"
    "import_SED" "import_SKB" "import_SRB" "import_TYB" "import_UKC"
)

for cmd in "${IMPORT_COMMANDS[@]}"; do
    echo "Importing $cmd..."
    python manage.py "$cmd" || echo "Warning: $cmd failed"
done

echo "Starting server..."
python manage.py runserver 0.0.0.0:8000
