cleanup:
	rm tools/*.ipynb
	rm -r tools/__pycache__

cleanall: cleanup
	rm assets
	rm -r tools/camera_calibration_images
