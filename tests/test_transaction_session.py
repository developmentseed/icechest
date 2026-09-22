import icechunk, pytest, zarr
from icechest import HybridRepo


def test_transaction_adopts_supplied_session(tmp_path):
    repo = HybridRepo.create(
        icechunk.local_filesystem_storage(str(tmp_path / "ic")),
        str(tmp_path / "wh"),
    )
    session = repo.repo.writable_session("main")
    zarr.open_group(session.store, path="/staged", mode="a").attrs["k"] = "v"

    tx = repo.transaction("main", "adopted", session=session)
    assert tx.session is session
    snapshot = tx.commit()

    group = repo.read(snapshot_id=snapshot).group
    assert group["staged"].attrs["k"] == "v"


def test_transaction_rejects_session_on_another_branch(tmp_path):
    repo = HybridRepo.create(
        icechunk.local_filesystem_storage(str(tmp_path / "ic")),
        str(tmp_path / "wh"),
    )
    session = repo.repo.writable_session("main")
    with pytest.raises(ValueError, match="branch"):
        repo.transaction("other", "mismatched", session=session)
