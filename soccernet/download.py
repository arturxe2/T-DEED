from SoccerNet.Downloader import SoccerNetDownloader

# mySoccerNetDownloader = SoccerNetDownloader(LocalDirectory="labels")
# mySoccerNetDownloader.downloadGames(files=["Labels-v2.json"], split=["test"])



mySoccerNetDownloader = SoccerNetDownloader(LocalDirectory="videos")

mySoccerNetDownloader.password = "s0cc3rn3t"
mySoccerNetDownloader.downloadGames(files=["1_224p.mkv", "2_224p.mkv"], split=["test"])
