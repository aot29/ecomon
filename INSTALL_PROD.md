# Installing an Ecomon instance for development
Instructions for installing Ecomon on a Linux server.

System requirements for production:
* 250GB+ RAM
* 25+ CPU cores
* 2+ GPU
* 8TB on a fast disk to hold the database
* 70TB+ to store audio data and for backups, e.g. on a 3fs storage
* Sudo access

Required software
* CUDA Version: 12.4
* Git
* Docker and Docker-Compose

Clone the repository into the desired directory (ecomon_SITENAME_MODELNAME)
```
sudo GIT_SSH_COMMAND='ssh -i path-to-your-id_rsa' git clone git@github.com:MfN-Berlin/ecomon.git ecomon_SITENAME_MODELNAME
sudo chown -R "$USER":akwamo ecomon_SITENAME_MODELNAME
cd ecomon_SITENAME_MODELNAME
```
Check that you are on the correct branch: `git status` should give "new-main"

If you need to install multiple instances on a single (virtual) machine, you will need additional steps, see under: "Multiple instances" below.

Make a copy of the environment variables file for development
```
cp env-production .env
```

Open the `.env` file in an editor and set at least:
* DB_PASSWORD=secure-password
* DB_ROOT_PASSWORD=secure-password
* BASE_DATA_DIRECTORY=path-to-audio-dir  # needs 50TB+, e.g. on 3fs storage
* PGBACKUP_PATH=path-to-backup-dir  # needs 20TB+, e.g. on 3fs storage
* DOMAIN=your-domain-or-ip
* SUB_PATH=/your-subpath  # if you change this, then you will have to rebuild the frotend container
* PGDATA_PATH=path-to-custom-place-for-database-data  # important: db needs 8TB+ on fast disk
* AIRFLOW_ADMIN_PASSWORD=secure-password
* HASURA_ADMIN_SECRET=secure-password
* HASURA_URL=pdefault-docker-compose-gateway (typically 172.17.0.1, used by Dashboard)
* USE_GPU=1  # 1, 2, or all
* TMP_DIR=/mnt/akwamotmp/your-subpath # this should be unique, and should exist
* PGBACKUP_PATH=/mnt/akwamodb/ecomon_BRITZ01_glob11k # this should be unique, and should exist, and by convention the name should end with "_backup"

Make sure that these are the same in .env and/or docker-compose.production.yaml:
* The port in .env HASURA_URL should be unique and the port exposed by the graphql-engine service in docker.compose.production.yaml
* Port to dashboard service
* Port to DB service should be unique

Check that PGDATA_PATH exista and contains wav data.
Check that TMP_DIR, PGBACKUP_PATH exist, is not used by another instance, or else create it.

Give the redis service in docker-compose.production.yaml a unique name

In .env, set the ENTRY_PORT to the port this instance of ecomon should listen to. This is the entry port that your reverse-proxy (which might be on another machine) is mapping the URL https://your-ip-adress/SUB_PATH to.

Start Docker containers
```
docker compose -f docker-compose.production.yaml up -d
```
This should start these containers:
* ecomon_validate-dashboard-1
* ecomon_validate-airflow-1
* ecomon_validate-db-migrate-1
* ecomon_validate-api-1
* ecomon_validate-traefik-1
* ecomon_validate-worker-1
* ecomon_validate-frontend-1
* redis-validate
* ecomon_validate-graphql-engine-1
* ecomon_validate-db-1

And these networks:
* ecomon_validate_ecomon
* ecomon_validate_traefik-ingress

Restore the sql schema from backup
`psql -U your_username -d target_database_name -f output_schema.sql`


## Multiple instances
If you need multiple instances of ecomon, then the recommended solution is to put each instance in its own virtual machine.

However, if you need to install multiple instances on the same (virtual) machine, you will need to make a local fork of the repository, and rebuild the frontend container.

