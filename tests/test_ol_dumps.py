from __future__ import annotations

import gzip
import json
from pathlib import Path

import httpx
import pytest

from pipeline.sources import ol_dumps as dumps


def _gz(path: Path, lines: list[str]) -> Path:
    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path


def _work_line(key: str, **record: object) -> str:
    return "\t".join(["/type/work", f"/works/{key}", "3", "2024-01-01", json.dumps(record)])


def _author_line(key: str, name: str) -> str:
    return "\t".join(
        ["/type/author", f"/authors/{key}", "1", "2024-01-01", json.dumps({"name": name})]
    )


@pytest.fixture
def works_dump(tmp_path: Path) -> Path:
    return _gz(
        tmp_path / "works.gz",
        [
            _work_line(
                "OL1W",
                title="First",
                description={"type": "/type/text", "value": "A long enough description here."},
                subjects=["Fiction", 7, "Horror"],
                authors=[{"author": {"key": "/authors/OL9A"}}, {"type": "x"}],
                first_publish_date="Jan 28, 1977",
                covers=[-1, 555],
            ),
            _work_line("OL2W", title="Second", description="Plain text.\r\n----------\nSource"),
            _work_line("OL3W", title="Third"),
        ],
    )


def test_stream_records_parses_only_the_wanted_keys(works_dump: Path) -> None:
    got = dict(dumps.stream_records(works_dump, {"/works/OL1W", "/works/OL3W"}))
    assert set(got) == {"/works/OL1W", "/works/OL3W"}
    assert got["/works/OL1W"]["title"] == "First"


def test_stream_records_stops_once_everything_is_found(tmp_path: Path) -> None:
    path = _gz(tmp_path / "w.gz", [_work_line("OL1W", title="A")] + ["not a valid line"] * 5)
    assert [k for k, _ in dumps.stream_records(path, {"/works/OL1W"})] == ["/works/OL1W"]


def test_project_work_cleans_the_record(works_dump: Path) -> None:
    records = dict(dumps.stream_records(works_dump, {"/works/OL1W", "/works/OL2W", "/works/OL3W"}))
    first = dumps.project_work(records["/works/OL1W"])
    assert first["description"] == "A long enough description here."
    assert first["subjects"] == ["Fiction", "Horror"]  # non-strings dropped
    assert first["author_keys"] == ["/authors/OL9A"]
    assert first["cover_id"] == 555  # negative ids are placeholders
    assert dumps.project_work(records["/works/OL2W"])["description"] == "Plain text."
    assert dumps.project_work(records["/works/OL3W"])["description"] == ""


@pytest.mark.parametrize(
    ("value", "year"),
    [
        ("1977", 1977),
        ("Jan 28, 1977", 1977),
        ("c. 1850", 1850),
        ("", None),
        (None, None),
        ("n.d.", None),
    ],
)
def test_parse_year(value: object, year: int | None) -> None:
    assert dumps.parse_year(value) == year


def test_has_description_threshold() -> None:
    assert dumps.has_description({"description": "x" * 80}, 80)
    assert not dumps.has_description({"description": "x" * 79}, 80)
    assert not dumps.has_description(None, 80)


def test_count_reading_log_counts_every_shelf(tmp_path: Path) -> None:
    path = _gz(
        tmp_path / "log.gz",
        [
            "/works/OL1W\t\\N\tAlready Read\t2026-01-01",
            "/works/OL1W\t/books/OL5M\tWant to read\t2026-01-02",
            "/works/OL2W\t\\N\tCurrently Reading\t2026-01-03",
        ],
    )
    counts = dumps.count_reading_log(path)
    assert counts == {"/works/OL1W": 2, "/works/OL2W": 1}
    assert counts.most_common(1) == [("/works/OL1W", 2)]


def test_record_cache_streams_only_for_new_keys_and_remembers_misses(
    tmp_path: Path, works_dump: Path
) -> None:
    cache_file = tmp_path / "cache" / "works.jsonl"
    cache = dumps.RecordCache(cache_file, dumps.project_work)
    first = cache.get_many(["/works/OL1W", "/works/OL404W"], works_dump)
    assert first["/works/OL1W"]["title"] == "First" and first["/works/OL404W"] is None

    works_dump.unlink()  # the dump is gone; a fresh cache object must not need it
    again = dumps.RecordCache(cache_file, dumps.project_work)
    assert again.get_many(["/works/OL1W", "/works/OL404W"], works_dump) == first


def test_to_fetched_work_resolves_authors_and_popularity(works_dump: Path) -> None:
    record = dict(dumps.stream_records(works_dump, {"/works/OL1W"}))["/works/OL1W"]
    work = dumps.to_fetched_work(
        "/works/OL1W", dumps.project_work(record), {"/authors/OL9A": "Stephen King"}, 4200.0
    )
    assert (work.source, work.external_id, work.media_type) == ("openlibrary", "OL1W", "book")
    assert (work.title, work.year, work.creators) == ("First", 1977, ["Stephen King"])
    assert work.popularity_raw == 4200.0
    assert work.cover_url == "https://covers.openlibrary.org/b/id/555-M.jpg"


def test_author_projection_falls_back_to_personal_name() -> None:
    assert dumps.project_author({"name": "A. Writer"}) == {"name": "A. Writer"}
    assert dumps.project_author({"personal_name": "B. Writer"}) == {"name": "B. Writer"}
    assert dumps.project_author({}) == {"name": ""}


# Download ----------------------------------------------------------------------------------------

PAYLOAD = bytes(range(256)) * 40  # 10,240 bytes


def _server(payload: bytes = PAYLOAD, *, honour_range: bool = True) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        headers = {
            "content-length": str(len(payload)),
            "last-modified": "Fri, 01 Aug 2026 00:00:00 GMT",
        }
        if request.method == "HEAD":
            return httpx.Response(200, headers=headers)
        start = request.headers.get("range")
        if start and honour_range:
            offset = int(start.split("=")[1].rstrip("-"))
            if offset >= len(payload):
                return httpx.Response(416)
            return httpx.Response(206, content=payload[offset:])
        return httpx.Response(200, content=payload)

    return httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)


def test_download_writes_the_file_and_manifest(tmp_path: Path) -> None:
    path = dumps.download_dump("ratings", dest_dir=tmp_path, http=_server())
    assert path.read_bytes() == PAYLOAD
    manifest = json.loads((tmp_path / "MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["ratings"]["size"] == len(PAYLOAD)
    assert "2026" in manifest["ratings"]["last_modified"]
    assert not list(tmp_path.glob("*.part"))


def test_download_is_skipped_when_complete(tmp_path: Path) -> None:
    dumps.download_dump("ratings", dest_dir=tmp_path, http=_server())
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.method)
        return httpx.Response(200, headers={"content-length": str(len(PAYLOAD))})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    dumps.download_dump("ratings", dest_dir=tmp_path, http=client)
    assert calls == ["HEAD"]  # no body request


def test_download_resumes_a_partial_file(tmp_path: Path) -> None:
    part = tmp_path / "ol_dump_ratings_latest.txt.gz.part"
    part.write_bytes(PAYLOAD[:4000])
    path = dumps.download_dump("ratings", dest_dir=tmp_path, http=_server())
    assert path.read_bytes() == PAYLOAD


def test_download_restarts_when_the_server_ignores_range(tmp_path: Path) -> None:
    (tmp_path / "ol_dump_ratings_latest.txt.gz.part").write_bytes(PAYLOAD[:4000])
    path = dumps.download_dump("ratings", dest_dir=tmp_path, http=_server(honour_range=False))
    assert path.read_bytes() == PAYLOAD


def test_download_replaces_a_stale_dump_of_a_different_size(tmp_path: Path) -> None:
    (tmp_path / "ol_dump_ratings_latest.txt.gz").write_bytes(b"old")
    path = dumps.download_dump("ratings", dest_dir=tmp_path, http=_server())
    assert path.read_bytes() == PAYLOAD


def test_download_refuses_when_the_disk_is_too_full(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Usage:
        free = 100

    monkeypatch.setattr(dumps.shutil, "disk_usage", lambda p: Usage)
    with pytest.raises(OSError, match="GB"):
        dumps.download_dump("ratings", dest_dir=tmp_path, http=_server())


def test_download_detects_a_truncated_transfer(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "HEAD":
            return httpx.Response(200, headers={"content-length": str(len(PAYLOAD))})
        return httpx.Response(200, content=PAYLOAD[:100])

    client = httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(OSError, match="expected"):
        dumps.download_dump("ratings", dest_dir=tmp_path, http=client)
    assert (tmp_path / "ol_dump_ratings_latest.txt.gz.part").exists()  # kept for resuming


# Segmented (parallel) download -----------------------------------------------------------------


def _range_server(payload: bytes = PAYLOAD, *, accept_ranges: bool = True, honour: bool = True):
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        headers = {"content-length": str(len(payload))}
        if accept_ranges:
            headers["accept-ranges"] = "bytes"
        if request.method == "HEAD":
            return httpx.Response(200, headers=headers)
        spec = request.headers.get("range")
        if spec:
            seen.append(spec)
        if spec and honour:
            start, _, end = spec.removeprefix("bytes=").partition("-")
            stop = int(end) + 1 if end else len(payload)
            return httpx.Response(206, content=payload[int(start) : stop])
        return httpx.Response(200, content=payload)

    return httpx.Client(transport=httpx.MockTransport(handler)), seen


def test_segments_cover_the_file_exactly_once() -> None:
    for total, count in [(10_240, 4), (10, 3), (7, 10), (1, 1)]:
        ranges = dumps._segments(total, count)
        assert ranges[0][0] == 0 and ranges[-1][1] == total - 1
        assert all(b[0] == a[1] + 1 for a, b in zip(ranges, ranges[1:], strict=False))


def test_a_segmented_download_joins_the_pieces_in_order(tmp_path: Path) -> None:
    client, seen = _range_server()
    path = dumps.download_dump(
        "works", dest_dir=tmp_path, http=client, connections=4, segment_threshold=1
    )
    assert path.read_bytes() == PAYLOAD
    assert len(seen) == 4  # one Range request per connection
    assert not list(tmp_path.glob("*.seg*")) and not list(tmp_path.glob("*.part"))
    assert json.loads((tmp_path / "MANIFEST.json").read_text(encoding="utf-8"))["works"][
        "size"
    ] == len(PAYLOAD)


def test_a_segmented_download_resumes_a_partial_segment(tmp_path: Path) -> None:
    (first_start, first_end), *_ = dumps._segments(len(PAYLOAD), 4)
    done = 1000
    (tmp_path / "ol_dump_works_latest.txt.gz.seg0of4").write_bytes(
        PAYLOAD[first_start : first_start + done]
    )
    client, seen = _range_server()
    path = dumps.download_dump(
        "works", dest_dir=tmp_path, http=client, connections=4, segment_threshold=1
    )
    assert path.read_bytes() == PAYLOAD
    assert f"bytes={first_start + done}-{first_end}" in seen  # asked only for the missing part


def test_segment_files_from_a_different_connection_count_are_discarded(tmp_path: Path) -> None:
    stale = tmp_path / "ol_dump_works_latest.txt.gz.seg0of8"
    stale.write_bytes(b"junk")
    client, _ = _range_server()
    dumps.download_dump("works", dest_dir=tmp_path, http=client, connections=4, segment_threshold=1)
    assert not stale.exists()


def test_it_falls_back_to_one_connection_when_ranges_are_not_offered(tmp_path: Path) -> None:
    client, seen = _range_server(accept_ranges=False)
    path = dumps.download_dump(
        "works", dest_dir=tmp_path, http=client, connections=4, segment_threshold=1
    )
    assert path.read_bytes() == PAYLOAD and seen == []


def test_small_files_use_one_connection_even_when_more_are_allowed(tmp_path: Path) -> None:
    client, seen = _range_server()
    dumps.download_dump("works", dest_dir=tmp_path, http=client, connections=4)  # default threshold
    assert seen == []


def test_a_server_that_ignores_range_fails_loudly_and_keeps_finished_segments(
    tmp_path: Path,
) -> None:
    client, _ = _range_server(honour=False)
    with pytest.raises(OSError, match="Range"):
        dumps.download_dump(
            "works", dest_dir=tmp_path, http=client, connections=4, segment_threshold=1
        )
    assert not (tmp_path / "ol_dump_works_latest.txt.gz").exists()


def test_the_segmented_path_checks_disk_for_the_pieces_and_the_joined_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Usage:
        free = len(PAYLOAD) + (1 << 30)  # room for one copy, not for two

    monkeypatch.setattr(dumps.shutil, "disk_usage", lambda p: Usage)
    client, _ = _range_server()
    with pytest.raises(OSError, match="GB"):
        dumps.download_dump(
            "works", dest_dir=tmp_path, http=client, connections=4, segment_threshold=1
        )
