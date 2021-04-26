
BACKUP_FOLDER="/PAGER/WP8/docker_images_backup/"

WP2GONG=wp2-get_magnetogram
WP2AWSOM=wp2-swmf
WP2SWIFT=wp2-swift
WP3SWPC=wp3-swpc-kp
WP3NIEMEGK=wp3-niemegk-kp
WP3SWIFTKP=wp3-swift-kp
WP3L1=wp3-l1-geoindexes

result=$( docker images -q $WP2GONG )
if [[ -n "$result" ]]; then
  echo "Image $WP2GONG Found. Backing up"
  docker save $WP2GONG > $BACKUP_FOLDER/wp2-gong.tar
fi

result=$( docker images -q $WP2AWSOM )
if [[ -n "$result" ]]; then
  echo "Image $WP2AWSOM Found. Backing up"
  docker save $WP2AWSOM > $BACKUP_FOLDER/wp2-awsom.tar
fi

result=$( docker images -q $WP2SWIFT )
if [[ -n "$result" ]]; then
  echo "Image $WP2SWIFT Found. Backing up"
  docker save $WP2SWIFT > $BACKUP_FOLDER/wp2-swift.tar
fi

result=$( docker images -q $WP3SWPC )
if [[ -n "$result" ]]; then
  echo "Image $WP3SWPC Found. Backing up"
  docker save $WP3SWPC > $BACKUP_FOLDER/wp3-swpc.tar
fi

result=$( docker images -q $WP3NIEMEGK )
if [[ -n "$result" ]]; then
  echo "Image $WP3NIEMEGK Found. Backing up"
  docker save $WP3NIEMEGK > $BACKUP_FOLDER/wp3-niemegk.tar
fi

result=$( docker images -q $WP3SWIFTKP )
if [[ -n "$result" ]]; then
  echo "Image $WP3SWIFTKP Found. Backing up"
  docker save $WP3SWIFTKP > $BACKUP_FOLDER/wp3-swift-kp.tar
fi

result=$( docker images -q $WP3L1 )
if [[ -n "$result" ]]; then
  echo "Image $WP3L1 Found. Backing up"
  docker save $WP3L1 > $BACKUP_FOLDER/wp3-l1.tar
fi
