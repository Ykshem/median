-- Demo personas and their pre-rated works.
--
-- Safe to run on every init_db() and again after the catalog is built (the index step does):
--   * personas use fixed ids + INSERT OR IGNORE;
--   * shelf items reference works by EXTERNAL id (TMDB movie id / Open Library work id), not by
--     database id, so they survive catalog rebuilds. A work that isn't in the catalog yet simply
--     produces no row, and appears the next time this file runs.
-- INSERT OR IGNORE never overwrites a rating the user has since changed.

INSERT OR IGNORE INTO personas (id, name, description) VALUES
  (1, 'Horror fan',          'Loves slow-burn dread, haunted places and unreliable narrators.'),
  (2, 'Cozy mystery reader', 'Likes small-town puzzles, warm tea and gentle stakes.'),
  (3, 'Epic fantasy fan',    'Wants sprawling worlds, long quests and found family.'),
  (4, 'Literary drama',      'Drawn to quiet, character-driven stories about grief and family.');

WITH seed(persona_id, source, external_id, shelf, rating) AS (VALUES
  -- 1  Horror fan
  (1, 'tmdb',        '493922',      'favorite', 10),  -- Hereditary
  (1, 'tmdb',        '694',         'finished',  9),  -- The Shining (film)
  (1, 'tmdb',        '348',         'finished',  9),  -- Alien
  (1, 'tmdb',        '242224',      'finished',  8),  -- The Babadook
  (1, 'tmdb',        '419430',      'finished',  8),  -- Get Out
  (1, 'openlibrary', 'OL20759125W', 'favorite', 10),  -- Mexican Gothic
  (1, 'openlibrary', 'OL81626W',    'finished',  8),  -- Carrie
  (1, 'openlibrary', 'OL20893680W', 'finished',  7),  -- Piranesi
  (1, 'tmdb',        '391713',      'finished',  3),  -- Lady Bird (not for them)
  -- 2  Cozy mystery reader
  (2, 'tmdb',        '546554',      'favorite', 10),  -- Knives Out
  (2, 'openlibrary', 'OL20878311W', 'favorite',  9),  -- The Thursday Murder Club
  (2, 'tmdb',        '4176',        'finished',  9),  -- Murder on the Orient Express (1974)
  (2, 'openlibrary', 'OL471576W',   'finished',  8),  -- Murder on the Orient Express (book)
  (2, 'tmdb',        '120467',      'finished',  9),  -- The Grand Budapest Hotel
  (2, 'openlibrary', 'OL17797130W', 'finished',  8),  -- A Gentleman in Moscow
  (2, 'openlibrary', 'OL20793817W', 'finished',  8),  -- Anxious People
  (2, 'tmdb',        '2493',        'finished',  8),  -- The Princess Bride (film)
  (2, 'openlibrary', 'OL483391W',   'finished',  7),  -- Charlotte's Web
  (2, 'tmdb',        '493922',      'finished',  2),  -- Hereditary (far too scary)
  -- 3  Epic fantasy fan
  (3, 'tmdb',        '120',         'favorite', 10),  -- The Fellowship of the Ring (film)
  (3, 'openlibrary', 'OL8479867W',  'favorite', 10),  -- The Name of the Wind
  (3, 'openlibrary', 'OL27482W',    'finished',  9),  -- The Hobbit
  (3, 'tmdb',        '438631',      'finished',  9),  -- Dune (2021)
  (3, 'tmdb',        '129',         'finished',  9),  -- Spirited Away
  (3, 'openlibrary', 'OL21213411W', 'finished',  8),  -- The Bear and the Nightingale
  (3, 'openlibrary', 'OL16086747W', 'finished',  8),  -- The Night Circus
  (3, 'openlibrary', 'OL18012166W', 'finished',  8),  -- Circe
  (3, 'tmdb',        '671',         'finished',  8),  -- Harry Potter and the Philosopher's Stone
  (3, 'tmdb',        '76',          'finished',  3),  -- Before Sunrise (too talky)
  -- 4  Literary drama
  (4, 'openlibrary', 'OL1875009W',  'favorite', 10),  -- Gilead
  (4, 'tmdb',        '376867',      'favorite', 10),  -- Moonlight
  (4, 'openlibrary', 'OL59048W',    'finished',  9),  -- The Remains of the Day (book)
  (4, 'tmdb',        '1245',        'finished',  9),  -- The Remains of the Day (film)
  (4, 'tmdb',        '391713',      'finished',  9),  -- Lady Bird
  (4, 'openlibrary', 'OL20150260W', 'finished',  8),  -- Normal People
  (4, 'openlibrary', 'OL17762217W', 'finished',  8),  -- Pachinko
  (4, 'openlibrary', 'OL20883297W', 'finished',  8),  -- Klara and the Sun
  (4, 'tmdb',        '42188',       'finished',  7),  -- Never Let Me Go (film)
  (4, 'openlibrary', 'OL50548W',    'want',   NULL),  -- Beloved
  (4, 'tmdb',        '76341',       'finished',  3)   -- Mad Max: Fury Road (not for them)
)
INSERT OR IGNORE INTO shelf_items (persona_id, work_id, shelf, rating)
SELECT s.persona_id, e.work_id, s.shelf, s.rating
FROM seed s
JOIN external_ids e ON e.source = s.source AND e.external_id = s.external_id;
