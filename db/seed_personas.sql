-- Demo personas (rows only). Pre-rated works are added in M2, once a catalog exists.
-- Fixed ids + INSERT OR IGNORE keep this safe to run on every init_db().
INSERT OR IGNORE INTO personas (id, name, description) VALUES
  (1, 'Horror fan',          'Loves slow-burn dread, haunted places and unreliable narrators.'),
  (2, 'Cozy mystery reader', 'Likes small-town puzzles, warm tea and gentle stakes.'),
  (3, 'Epic fantasy fan',    'Wants sprawling worlds, long quests and found family.'),
  (4, 'Literary drama',      'Drawn to quiet, character-driven stories about grief and family.');
