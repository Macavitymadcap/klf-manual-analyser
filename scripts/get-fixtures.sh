!#/bin/bash

fixtures = "tests/fixtures"

mkdir -p $fixtures

# Bessie Smith
curl -L -o "$fixtures/Bessie_Smith-Nobody_Knows_You_When_Youre_Down_And_Out.mp3" \
  "https://archive.org/download/78_nobody-knows-you-when-youre-down-and-out_bessie-smith-edward-allen-cyrus-st-clair-c_gbia3032633a/NOBODY%20KNOWS%20YOU%20WHEN%20YOU%27RE%20DOWN%20AND%20OUT.mp3"

curl -L -o "$fixtures/Bessie_Smith-Back_Water_Blues.mp3" \
  "https://archive.org/download/78_back-water-blues_bessie-smith-james-p-johnson-smith_gbia3032633b/BACK%20WATER%20BLUES%20-%20BESSIE%20SMITH%20-%20James%20P.%20Johnson.mp3"

curl -L -o "$fixtures/Bessie_Smith-Gimmie_A_Pigfoot.mp3" \
  "https://archive.org/download/78_gimmie-a-pigfoot_bessie-smith-f-newton-j-teagarden-l-berry-b-goodman-b-washington-b_gbia3037975a/GIMMIE%20A%20PIGFOOT%20-%20BESSIE%20SMITH%20-%20F.%20Newton.mp3"

curl -L -o "$fixtures/Bessie_Smith-Take_Me_For_A_Buggy_Ride.mp3" \
  "https://archive.org/download/78_take-me-for-a-buggy-ride_bessie-smith-wilson_gbia3041099b/TAKE%20ME%20FOR%20A%20BUGGY%20RIDE%20-%20BESSIE%20SMITH%20-%20Wilson.mp3"

# Louis Armstrong Hot Five
curl -L -o "$fixtures/Louis_Armstrong-Heebie_Jeebies.mp3" \
  "https://archive.org/download/louis-armstrong-louis-armstrong-1923-1927/05Louis%20Armstrong%20%26%20His%20Hot%20Five%20%E2%80%93%20Heebie%20Jeebies.mp3"

# Jelly Roll Morton
curl -L -o "$fixtures/Jelly_Roll_Morton-Black_Bottom_Stomp.mp3" \
  "https://archive.org/download/78_black-bottom-stomp_jelly-roll-mortons-red-hot-morton_gbia0076785a/Black%20Bottom%20Stomp%20-%20Jelly%20Roll%20Morton%27s%20Red%20Hot.mp3"

echo "Fixtures downloaded to $fixtures"