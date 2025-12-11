"""
ULTRA PRECISE FRUIT SCRAPER v3.0
Usa tecnicas avanzadas de todas las APIs para precision maxima:
- Pixabay: exclusiones con '-' + category='food'
- Unsplash: collections curadas + filtros de color
- Pexels: multiples queries + validacion estricta
"""

import requests
import time
import random
import shutil
from pathlib import Path
import sys
sys.path.append('..')
from src.utils import _validate_source_integrity

try:
    from PIL import Image
except ImportError:
    print("Error: PIL not installed. Run: pip install pillow")
    exit(1)


class UltraPreciseScraper:
    """
    Scraper de precision maxima para frutas.
    """

    def __init__(self, output_dir='data_fruits'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.unsplash_key = "51IPjnUcI1MM8-qksh0YU8-wOdwUKYt5V_Hnd___00c"
        self.pexels_key = "1VeX6CBkGcu7887YUxuxTx6RKQSDVLd3jqzIp8z66Qr4vnBiiRN64k48"
        self.pixabay_key = "53668996-ca552ff2753fd08bcc9a46f67"

        self.fruits360_mapping = {
            'Albaricoques': ['Apricot 1'],
            'Higos': [],
            'Ciruelas': ['Plum 1', 'Plum 2', 'Plum 3'],
            'Cerezas': ['Cherry 1', 'Cherry 2', 'Cherry Rainier 1', 'Cherry Wax Black 1', 'Cherry Wax Red 1', 'Cherry Wax Yellow 1'],
            'Melón': ['Cantaloupe 1', 'Cantaloupe 2', 'Melon Piel de Sapo 1'],
            'Sandía': ['Watermelon 1'],
            'Nectarinas': ['Nectarine 1', 'Nectarine Flat 1'],
            'Paraguayos': ['Peach Flat 1'],
            'Melocotón': ['Peach 1', 'Peach 2'],
            'Nísperos': ['Loquat 1'],
            'Pera': ['Pear 1', 'Pear 2', 'Pear Abate 1', 'Pear Forelle 1', 'Pear Kaiser 1', 'Pear Monster 1', 'Pear Red 1', 'Pear Stone 1', 'Pear Williams 1'],
            'Plátano': ['Banana 1', 'Banana 3', 'Banana 4', 'Banana Lady Finger 1', 'Banana Red 1'],
            'Frutos rojos': ['Strawberry 1', 'Raspberry 1', 'Blueberry 1', 'Blackberrie 1', 'Blackberrie 2'],
            'Caqui': ['Persimmon 1'],
            'Chirimoya': [],
            'Granada': ['Pomegranate 1'],
            'Kiwis': ['Kiwi 1'],
            'Mandarinas': ['Mandarine 1', 'Clementine 1', 'Tangelo 1'],
            'Manzana': ['Apple 10', 'Apple 11', 'Apple 12', 'Apple 13', 'Apple 14', 'Apple 17', 'Apple 18', 'Apple 19',
                        'Apple 5', 'Apple 6', 'Apple 7', 'Apple 8', 'Apple 9', 'Apple Braeburn 1', 'Apple Golden 1',
                        'Apple Golden 2', 'Apple Golden 3', 'Apple Granny Smith 1', 'Apple Pink Lady 1', 'Apple Red 1',
                        'Apple Red 2', 'Apple Red 3', 'Apple Red Delicious 1', 'Apple Red Yellow 1', 'Apple Red Yellow 2'],
            'Naranja': ['Orange 1'],
            'Pomelo': ['Grapefruit 1', 'Grapefruit Pink 1']
        }

        self.fruit_mapping = {
            'Albaricoques': {
                'en': 'apricot',
                'queries': ['apricot fruit', 'fresh apricot', 'apricot whole'],
                'keywords': ['apricot'],
                'exclude': ['peach', 'nectarine', 'plum'],
                'color': 'orange',
                'pixabay_exclude': ['flower', 'blossom', 'bloom', 'tree', 'branch']
            },
            'Higos': {
                'en': 'fig',
                'queries': ['fig fruit', 'fresh fig', 'whole fig'],
                'keywords': ['fig'],
                'exclude': ['date', 'prune'],
                'color': None,
                'pixabay_exclude': ['flower', 'blossom', 'tree']
            },
            'Ciruelas': {
                'en': 'plum',
                'queries': ['plum fruit', 'fresh plum', 'whole plum'],
                'keywords': ['plum'],
                'exclude': ['prune', 'apricot'],
                'color': None,
                'pixabay_exclude': ['flower', 'blossom', 'tree']
            },
            'Cerezas': {
                'en': 'cherry',
                'queries': ['cherry fruit', 'fresh cherry', 'cherries'],
                'keywords': ['cherry', 'cherries'],
                'exclude': ['berry', 'strawberry'],
                'color': 'red',
                'pixabay_exclude': ['flower', 'blossom', 'bloom', 'tree']
            },
            'Melón': {
                'en': 'melon',
                'queries': ['melon fruit', 'cantaloupe', 'honeydew melon'],
                'keywords': ['melon', 'cantaloupe', 'honeydew'],
                'exclude': ['watermelon'],
                'color': 'green',
                'pixabay_exclude': ['watermelon']
            },
            'Sandía': {
                'en': 'watermelon',
                'queries': ['watermelon fruit', 'fresh watermelon'],
                'keywords': ['watermelon'],
                'exclude': ['melon'],
                'color': 'red',
                'pixabay_exclude': ['melon']
            },
            'Nectarinas': {
                'en': 'nectarine',
                'queries': ['nectarine fruit', 'fresh nectarine'],
                'keywords': ['nectarine'],
                'exclude': ['peach', 'apricot'],
                'color': 'orange',
                'pixabay_exclude': ['peach', 'apricot', 'flower', 'blossom']
            },
            'Paraguayos': {
                'en': 'flat peach',
                'queries': ['flat peach', 'donut peach', 'paraguayo fruit'],
                'keywords': ['flat peach', 'donut peach', 'paraguayo'],
                'exclude': ['nectarine', 'apricot'],
                'color': 'orange',
                'pixabay_exclude': ['nectarine', 'apricot', 'flower']
            },
            'Melocotón': {
                'en': 'peach',
                'queries': ['peach fruit', 'fresh peach', 'whole peach'],
                'keywords': ['peach'],
                'exclude': ['nectarine', 'apricot', 'flat'],
                'color': 'orange',
                'pixabay_exclude': ['nectarine', 'apricot', 'flat', 'flower', 'blossom']
            },
            'Nísperos': {
                'en': 'loquat',
                'queries': ['loquat fruit', 'fresh loquat'],
                'keywords': ['loquat'],
                'exclude': [],
                'color': 'orange',
                'pixabay_exclude': ['flower', 'blossom', 'tree']
            },
            'Pera': {
                'en': 'pear',
                'queries': ['pear fruit', 'fresh pear', 'whole pear'],
                'keywords': ['pear'],
                'exclude': ['apple'],
                'color': 'green',
                'pixabay_exclude': ['apple', 'flower']
            },
            'Plátano': {
                'en': 'banana',
                'queries': ['banana fruit', 'fresh banana', 'bananas'],
                'keywords': ['banana'],
                'exclude': ['plantain'],
                'color': 'yellow',
                'pixabay_exclude': ['plantain', 'tree', 'plant']
            },
            'Frutos rojos': {
                'en': 'berries',
                'queries': ['mixed berries', 'fresh berries', 'berry fruit'],
                'keywords': ['berry', 'berries', 'strawberry', 'raspberry', 'blueberry'],
                'exclude': [],
                'color': 'red',
                'pixabay_exclude': ['flower']
            },
            'Caqui': {
                'en': 'persimmon',
                'queries': ['persimmon fruit', 'kaki fruit', 'fresh persimmon'],
                'keywords': ['persimmon', 'kaki'],
                'exclude': ['tomato'],
                'color': 'orange',
                'pixabay_exclude': ['tomato', 'flower', 'tree']
            },
            'Chirimoya': {
                'en': 'cherimoya',
                'queries': ['cherimoya fruit', 'custard apple'],
                'keywords': ['cherimoya', 'custard apple'],
                'exclude': [],
                'color': None,
                'pixabay_exclude': ['flower', 'tree']
            },
            'Granada': {
                'en': 'pomegranate',
                'queries': ['pomegranate fruit', 'fresh pomegranate'],
                'keywords': ['pomegranate'],
                'exclude': [],
                'color': 'red',
                'pixabay_exclude': ['flower', 'tree']
            },
            'Kiwis': {
                'en': 'kiwi',
                'queries': ['kiwi fruit', 'fresh kiwi', 'kiwifruit'],
                'keywords': ['kiwi', 'kiwifruit'],
                'exclude': [],
                'color': 'green',
                'pixabay_exclude': ['flower', 'plant']
            },
            'Mandarinas': {
                'en': 'mandarin',
                'queries': ['mandarin fruit', 'tangerine', 'clementine'],
                'keywords': ['mandarin', 'tangerine', 'clementine'],
                'exclude': ['orange', 'grapefruit'],
                'color': 'orange',
                'pixabay_exclude': ['orange', 'grapefruit']
            },
            'Manzana': {
                'en': 'apple',
                'queries': ['apple fruit', 'fresh apple', 'red apple'],
                'keywords': ['apple'],
                'exclude': ['pear'],
                'color': 'red',
                'pixabay_exclude': ['pear', 'flower', 'blossom']
            },
            'Naranja': {
                'en': 'orange',
                'queries': ['orange fruit', 'fresh orange', 'oranges'],
                'keywords': ['orange'],
                'exclude': ['mandarin', 'tangerine', 'grapefruit'],
                'color': 'orange',
                'pixabay_exclude': ['mandarin', 'tangerine', 'grapefruit']
            },
            'Pomelo': {
                'en': 'grapefruit',
                'queries': ['grapefruit fruit', 'fresh grapefruit'],
                'keywords': ['grapefruit'],
                'exclude': ['orange', 'lemon'],
                'color': None,
                'pixabay_exclude': ['orange', 'lemon']
            }
        }

        self.global_excludes = ['flower', 'blossom', 'bloom', 'flowering', 'blooming', 'tree', 'branch', 'leaf', 'leaves']
        self.session = requests.Session()
        self.download_stats = {fruit: 0 for fruit in self.fruit_mapping.keys()}

    def download_image(self, url, filepath):
        """
        Descarga y valida imagen.
        """
        try:
            response = self.session.get(url, timeout=10, stream=True)
            response.raise_for_status()

            content_type = response.headers.get('content-type', '')
            if not content_type.startswith('image/'):
                return False

            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            file_size = filepath.stat().st_size
            if file_size < 5000 or file_size > 20_000_000:
                filepath.unlink()
                return False

            try:
                img = Image.open(filepath)
                img.verify()
                img = Image.open(filepath)
                if img.size[0] < 100 or img.size[1] < 100:
                    filepath.unlink()
                    return False
            except Exception:
                filepath.unlink()
                return False

            return True
        except Exception:
            if filepath.exists():
                filepath.unlink()
            return False

    def validate_unsplash_metadata(self, photo, fruit_config):
        """
        Validacion estricta de metadatos Unsplash.
        """
        description = photo.get('description', '') or ''
        alt_desc = photo.get('alt_description', '') or ''
        tags = ' '.join([tag.get('title', '') for tag in photo.get('tags', [])])

        metadata = f"{description} {alt_desc} {tags}".lower()

        has_keyword = any(kw.lower() in metadata for kw in fruit_config['keywords'])
        all_excludes = fruit_config['exclude'] + self.global_excludes
        has_excluded = any(term.lower() in metadata for term in all_excludes)

        return has_keyword and not has_excluded

    def validate_pexels_metadata(self, photo, fruit_config):
        """
        Validacion de alt text de Pexels.
        """
        alt = (photo.get('alt', '') or '').lower()

        has_keyword = any(kw.lower() in alt for kw in fruit_config['keywords'])
        critical_excludes = ['flower', 'blossom', 'bloom', 'tree', 'branch']
        has_critical = any(term in alt for term in critical_excludes)

        return has_keyword and not has_critical

    def validate_pixabay_metadata(self, hit, fruit_config):
        """
        Validacion de tags de Pixabay.
        """
        tags = (hit.get('tags', '') or '').lower()
        tag_list = [t.strip() for t in tags.split(',')]

        has_keyword = any(
            any(kw.lower() in tag for tag in tag_list)
            for kw in fruit_config['keywords']
        )

        all_excludes = fruit_config['exclude'] + ['flower', 'blossom', 'bloom', 'tree']
        has_excluded = any(
            any(term.lower() in tag for tag in tag_list)
            for term in all_excludes
        )

        return has_keyword and not has_excluded

    def scrape_pixabay(self, fruit_es, fruit_config, max_images=150):
        """
        Scrape Pixabay con exclusiones directas en query.
        PRIORIDAD 1: Mayor precision.
        """
        print(f"  [PIXABAY] Scraping {fruit_es}...")

        fruit_dir = self.output_dir / fruit_es
        fruit_dir.mkdir(exist_ok=True)

        downloaded = 0
        page = 1
        per_page = 200

        exclude_terms = fruit_config['pixabay_exclude']
        exclude_str = ' '.join([f'-{term}' for term in exclude_terms])
        query = f"{fruit_config['en']} fruit {exclude_str}"

        print(f"    Query: '{query}'")

        while downloaded < max_images:
            try:
                url = "https://pixabay.com/api/"
                params = {
                    'key': self.pixabay_key,
                    'q': query,
                    'category': 'food',
                    'image_type': 'photo',
                    'page': page,
                    'per_page': per_page
                }

                if fruit_config['color']:
                    params['colors'] = fruit_config['color']

                response = self.session.get(url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()

                if not data.get('hits'):
                    break

                for hit in data['hits']:
                    if downloaded >= max_images:
                        break

                    if not self.validate_pixabay_metadata(hit, fruit_config):
                        continue

                    img_url = hit['webformatURL']
                    photo_id = hit['id']
                    filepath = fruit_dir / f"pixabay_{photo_id}.jpg"

                    if filepath.exists():
                        continue

                    if self.download_image(img_url, filepath):
                        downloaded += 1
                        if downloaded % 10 == 0 or downloaded == max_images:
                            print(f"    [PIXABAY] {fruit_es}: {downloaded}/{max_images}")

                    time.sleep(0.2)

                page += 1

            except Exception as e:
                print(f"  [PIXABAY] Error for {fruit_es}: {e}")
                break

        return downloaded

    def scrape_unsplash(self, fruit_es, fruit_config, max_images=105):
        """
        Scrape Unsplash con collections curadas y multiples queries.
        PRIORIDAD 2: Alta calidad.
        """
        print(f"  [UNSPLASH] Scraping {fruit_es}...")

        fruit_dir = self.output_dir / fruit_es
        fruit_dir.mkdir(exist_ok=True)

        downloaded = 0
        collections = '9434066,1117930,191435'

        queries = fruit_config.get('queries', [fruit_config['en']])

        for query in queries:
            if downloaded >= max_images:
                break

            page = 1
            per_page = 30

            while downloaded < max_images:
                try:
                    url = "https://api.unsplash.com/search/photos"
                    params = {
                        'query': query,
                        'page': page,
                        'per_page': per_page,
                        'client_id': self.unsplash_key,
                        'collections': collections,
                        'orientation': 'squarish'
                    }

                    if fruit_config['color']:
                        params['color'] = fruit_config['color']

                    response = self.session.get(url, params=params, timeout=10)
                    response.raise_for_status()
                    data = response.json()

                    if not data.get('results'):
                        break

                    for photo in data['results']:
                        if downloaded >= max_images:
                            break

                        if not self.validate_unsplash_metadata(photo, fruit_config):
                            continue

                        img_url = photo['urls']['regular']
                        photo_id = photo['id']
                        filepath = fruit_dir / f"unsplash_{photo_id}.jpg"

                        if filepath.exists():
                            continue

                        if self.download_image(img_url, filepath):
                            downloaded += 1
                            if downloaded % 10 == 0 or downloaded == max_images:
                                print(f"    [UNSPLASH] {fruit_es}: {downloaded}/{max_images}")

                        time.sleep(0.5)

                    page += 1

                except Exception as e:
                    print(f"  [UNSPLASH] Error [{query}]: {e}")
                    break

        return downloaded

    def scrape_pexels(self, fruit_es, fruit_config, max_images=45):
        """
        Scrape Pexels con multiples queries.
        PRIORIDAD 3: Complemento.
        """
        print(f"  [PEXELS] Scraping {fruit_es}...")

        fruit_dir = self.output_dir / fruit_es
        fruit_dir.mkdir(exist_ok=True)

        downloaded = 0
        queries = fruit_config.get('queries', [fruit_config['en']])

        for query in queries:
            if downloaded >= max_images:
                break

            page = 1
            per_page = 80

            while downloaded < max_images:
                try:
                    url = "https://api.pexels.com/v1/search"
                    headers = {'Authorization': self.pexels_key}
                    params = {
                        'query': query,
                        'page': page,
                        'per_page': per_page,
                        'orientation': 'square',
                        'size': 'medium'
                    }

                    if fruit_config['color']:
                        params['color'] = fruit_config['color']

                    response = self.session.get(url, headers=headers, params=params, timeout=10)
                    response.raise_for_status()
                    data = response.json()

                    if not data.get('photos'):
                        break

                    for photo in data['photos']:
                        if downloaded >= max_images:
                            break

                        if not self.validate_pexels_metadata(photo, fruit_config):
                            continue

                        img_url = photo['src']['medium']
                        photo_id = photo['id']
                        filepath = fruit_dir / f"pexels_{photo_id}.jpg"

                        if filepath.exists():
                            continue

                        if self.download_image(img_url, filepath):
                            downloaded += 1
                            if downloaded % 10 == 0 or downloaded == max_images:
                                print(f"    [PEXELS] {fruit_es}: {downloaded}/{max_images}")

                        time.sleep(0.3)

                    page += 1

                except Exception as e:
                    print(f"  [PEXELS] Error [{query}]: {e}")
                    break

        return downloaded


    def scrape_all(self, images_per_fruit=1200):
        """
        Scrape todas las frutas con precision maxima.
        """
        print("=" * 70)
        print("ULTRA PRECISE FRUIT SCRAPER v3.0")
        print("=" * 70)
        print(f"Target: {images_per_fruit} images per fruit")
        print(f"Total fruits: {len(self.fruit_mapping)}")
        print("\nDistribucion:")
        print("  - Pixabay: 50% (exclusiones + category=food)")
        print("  - Unsplash: 35% (collections curadas + filtros)")
        print("  - Pexels: 15% (complemento)")
        print("=" * 70)

        pixabay_per_fruit = int(images_per_fruit * 0.5)
        unsplash_per_fruit = int(images_per_fruit * 0.35)
        pexels_per_fruit = int(images_per_fruit * 0.15)

        for fruit_es, fruit_config in self.fruit_mapping.items():
            print(f"\n{'=' * 70}")
            print(f"Downloading: {fruit_es} ({fruit_config['en']})")
            print(f"{'=' * 70}")

            total_downloaded = 0

            if self.pixabay_key:
                count = self.scrape_pixabay(fruit_es, fruit_config, pixabay_per_fruit)
                total_downloaded += count

            if self.unsplash_key:
                count = self.scrape_unsplash(fruit_es, fruit_config, unsplash_per_fruit)
                total_downloaded += count

            if self.pexels_key:
                count = self.scrape_pexels(fruit_es, fruit_config, pexels_per_fruit)
                total_downloaded += count

            total_downloaded += _validate_source_integrity(
                self.fruits360_mapping,
                self.output_dir / fruit_es,
                self.fruits360_mapping.get(fruit_es, []),
                images_per_fruit
            )

            self.download_stats[fruit_es] = total_downloaded

            print(f"\nCompleted {fruit_es}: {total_downloaded} images")

            time.sleep(1)

        self.print_summary()

    def print_summary(self):
        """
        Imprime estadisticas de descarga.
        """
        print("\n" + "=" * 70)
        print("DOWNLOAD SUMMARY")
        print("=" * 70)

        for fruit, count in sorted(self.download_stats.items(), key=lambda x: x[1], reverse=True):
            print(f"  {fruit:20s}: {count:4d} images")

        total = sum(self.download_stats.values())
        print("=" * 70)
        print(f"TOTAL DOWNLOADED: {total} images")
        print(f"Average per fruit: {total / len(self.download_stats):.1f} images")
        print("=" * 70)


def main():
    """
    Ejecucion principal.
    """
    print("=" * 70)
    print("ULTRA PRECISE FRUIT SCRAPER v3.0")
    print("Con tecnicas avanzadas de precision maxima")
    print("=" * 70)
    print("\nAPI Keys configured")
    print("Target: 1200 images per fruit x 22 fruits = 26,400 total images")
    print("\nTecnicas aplicadas:")
    print("  - Pixabay: exclusiones con '-' + category='food'")
    print("  - Unsplash: collections curadas + filtros de color")
    print("  - Pexels: multiples queries + validacion estricta")
    print("=" * 70)

    scraper = UltraPreciseScraper(output_dir='data_fruits')
    images_per_fruit = 1200

    print(f"\nStarting download: {images_per_fruit} images x {len(scraper.fruit_mapping)} fruits")
    print("Estimated time: 2-3 hours with rate limiting")
    print(f"Saving to: {scraper.output_dir.absolute()}\n")

    scraper.scrape_all(images_per_fruit=images_per_fruit)

    print("\nScraping completed!")
    print(f"Images saved to: {scraper.output_dir.absolute()}")


if __name__ == "__main__":
    main()
