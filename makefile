.PHONY: dockerbuild
dockerbuild:
	docker compose build

.PHONY: dockerrun
dockerrun:
	bash -c "sudo docker compose up"

.PHONY: docker
docker:
	$(MAKE) dockerbuild
	$(MAKE) dockerrun

