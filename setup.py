# Copyright 2021 the Ithaca Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Setup module for Ithaca.

Only installs the inference components.

See README.md for more details.
"""

import pathlib
import setuptools

here = pathlib.Path(__file__).parent.resolve()
long_description = (here / 'README.md').read_text(encoding='utf-8')
setuptools.setup(
    name='ithaca',
    author='Ithaca team',
    author_email='deepmind-ithaca-team@google.com',
    version='0.1.0',
    license='Apache License, Version 2.0',
    description='Ithaca library for ancient text restoration and attribution.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=setuptools.find_packages(exclude=('train',)),
    package_data={'': ['*.txt']},
    install_requires=(here / 'requirements.txt').read_text().splitlines(),
    extras_require={
        'train': [
            'optax',
            'jaxline==0.0.5',
            'tensorflow-datasets',
        ]
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
