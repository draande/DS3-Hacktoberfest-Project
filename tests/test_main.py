from main import main


def test_main(capsys):
    main()
    captured = capsys.readouterr()
    assert "Hello from ds3-hacktoberfest-project!" in captured.out
