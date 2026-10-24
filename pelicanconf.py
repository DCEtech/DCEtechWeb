AUTHOR = 'Daniel'
SITENAME = 'DCEtech'
SITEURL = 'https://www.danielcejudo.com/'

THEME = 'theme/mytheme'
PATH = 'content'
STATIC_PATHS = ['images', 'css', 'extra/robots.txt']
# robots.txt se copia a la raíz de la web
EXTRA_PATH_METADATA = {'extra/robots.txt': {'path': 'robots.txt'}}

# Sitemap sin plugins: plantilla propia del tema renderizada como página directa.
# SITEMAP_BASE es absoluta a propósito: con RELATIVE_URLS, SITEURL se vuelve relativa
# dentro de las plantillas, y un sitemap exige URLs absolutas.
DIRECT_TEMPLATES = ['index', 'tags', 'categories', 'authors', 'archives', 'sitemap']
SITEMAP_SAVE_AS = 'sitemap.xml'
SITEMAP_BASE = 'https://www.danielcejudo.com'

TIMEZONE = 'Europe/Madrid'

DEFAULT_LANG = 'en'




# Feed generation is usually not desired when developing
FEED_ALL_ATOM = None
CATEGORY_FEED_ATOM = None
TRANSLATION_FEED_ATOM = None
AUTHOR_FEED_ATOM = None
AUTHOR_FEED_RSS = None

DEFAULT_PAGINATION = 10

RELATIVE_URLS = True