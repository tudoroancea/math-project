mkdir -p submission
rm -r submission/*
cp -r math-project-num-experiments/cstr/cstr_package/*.py submission/cstr/cstr_package
cp -r math-project-num-experiments/cstr/tests/*.py submission/cstr/tests
cp -r math-project-num-experiments/cstr/README.md submission/cstr
cp requirements.txt submission/cstr
cp math-project-report/math-project-report.pdf submission
zip -r submission.zip submission
