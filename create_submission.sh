mkdir -p submission/
rm -r submission/*

# copy the cstr dir
mkdir -p submission/cstr/tests
mkdir -p submission/cstr/cstr_package
cp -r math-project-num-experiments/cstr/cstr_package/*.py submission/cstr/cstr_package/
cp -r math-project-num-experiments/cstr/tests/*.py submission/cstr/tests/
cp -r math-project-num-experiments/cstr/README.md submission/cstr/
cp requirements.txt submission/cstr/

# copy the results
mkdir -p submission/results
cp -r math-project-report/csv/*.csv submission/results/
cp -r math-project-report/images/*.png submission/results/

# copy the report
cp math-project-report/math-project-report.pdf submission

# create a zip archive with everything
zip -r tudor_oancea_bachelor_project.zip submission
