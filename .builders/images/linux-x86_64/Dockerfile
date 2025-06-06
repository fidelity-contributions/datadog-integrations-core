ARG BASE_IMAGE=quay.io/pypa/manylinux2014_x86_64
FROM ${BASE_IMAGE}

ARG SOURCE_DATE_EPOCH

ENV SOURCE_DATE_EPOCH=${SOURCE_DATE_EPOCH}

# Script for the common task of fetching a source tarball, compiling and installing it
COPY install-from-source.sh /

ENV CFLAGS="-O2"
ENV CXXFLAGS="${CFLAGS}"
# Auditwheel will only set RPATH's for copied libs if an existing RPATH is found,
# so we may as well set it to origin here anyway.
# --strip-debug reduces binary sizes and improves reproducibility
ENV LDFLAGS="-Wl,-rpath,'\$\$ORIGIN' -Wl,--strip-debug"

# openssl
RUN yum install -y perl-IPC-Cmd perl-CPANPLUS && \
 cpanp -i List::Util 1.66 && \
 DOWNLOAD_URL="https://www.openssl.org/source/openssl-{{version}}.tar.gz" \
 VERSION="3.4.1" \
 SHA256="002a2d6b30b58bf4bea46c43bdd96365aaf8daa6c428782aa4feee06da197df3" \
 RELATIVE_PATH="openssl-{{version}}" \
 # https://docs.python.org/3/using/unix.html#custom-openssl
 INSTALL_COMMAND="make install_sw" \
 CONFIGURE_SCRIPT="./config" \
 bash install-from-source.sh \
 -fPIC shared \
 # This prevents the creation of dynamically loaded modules that would be problematic to bundle into Python wheels
 no-module \
 no-comp no-idea no-mdc2 no-rc5 no-ssl3 no-gost && \
 echo /usr/local/lib >> /etc/ld.so.conf.d/lib.conf && \
 echo /usr/local/ssl/lib >> /etc/ld.so.conf.d/openssl.conf && \
 ldconfig

# Compile and install Python 3
ENV PYTHON3_VERSION=3.12.10
RUN yum install -y libffi-devel && \
 DOWNLOAD_URL="https://python.org/ftp/python/{{version}}/Python-{{version}}.tgz" \
 VERSION="${PYTHON3_VERSION}" \
 SHA256="15d9c623abfd2165fe816ea1fb385d6ed8cf3c664661ab357f1782e3036a6dac" \
 RELATIVE_PATH="Python-{{version}}" \
 bash install-from-source.sh --prefix=/opt/python/${PYTHON3_VERSION} --with-ensurepip=yes --enable-ipv6 --with-dbmliborder=
ENV PATH="/opt/python/${PYTHON3_VERSION}/bin:${PATH}"
# Set up virtual environment for Python 3
RUN /opt/python/${PYTHON3_VERSION}/bin/python3 -m pip install --no-warn-script-location --upgrade pip \
 && /opt/python/${PYTHON3_VERSION}/bin/python3 -m pip install --no-warn-script-location virtualenv \
 && /opt/python/${PYTHON3_VERSION}/bin/python3 -m virtualenv /py3

# Rust toolchain (needed for cryptography)
ENV RUST_VERSION="1.76"
ENV RUSTC_SHA256="0b2f6c8f85a3d02fde2efc0ced4657869d73fccfce59defb4e8d29233116e6db"
ARG RUSTUP_VERSION="1.26.0"
ARG RUSTUP_SHA256="0b2f6c8f85a3d02fde2efc0ced4657869d73fccfce59defb4e8d29233116e6db"
RUN curl -sSL -o rustup-init "https://static.rust-lang.org/rustup/archive/${RUSTUP_VERSION}/x86_64-unknown-linux-gnu/rustup-init" \
 && echo "${RUSTUP_SHA256} rustup-init" | sha256sum --check \
 && chmod +x ./rustup-init \
 && ./rustup-init -y --profile minimal --default-toolchain "${RUST_VERSION}" \
 && echo "${RUSTC_SHA256}  $HOME/.cargo/bin/rustc" | sha256sum --check \
 && rm ./rustup-init
ENV PATH="/root/.cargo/bin:${PATH}"

# MQ Client library required by pymqi
ENV IBM_MQ_VERSION="9.2.4.0"
ENV IBM_MQ_SHA256="d0d583eba72daf20b3762976f8831c2e23150ace90509520e12f8cda5b5bdb49"
RUN mkdir -p /opt/mqm \
 && curl "https://s3.amazonaws.com/dd-agent-omnibus/ibm-mq-backup/${IBM_MQ_VERSION}-IBM-MQC-Redist-LinuxX64.tar.gz" -o /tmp/mq_client.tar.gz \
 && echo "${IBM_MQ_SHA256}  /tmp/mq_client.tar.gz" | sha256sum --check \
 && tar -C /opt/mqm -xf /tmp/mq_client.tar.gz \
 && rm -rf /tmp/mq_client.tar.gz

# krb5 for dependencies that require kerberos support
RUN \
 DOWNLOAD_URL="https://kerberos.org/dist/krb5/1.20/krb5-{{version}}.tar.gz" \
 VERSION="1.20.1" \
 SHA256="704aed49b19eb5a7178b34b2873620ec299db08752d6a8574f95d41879ab8851" \
 RELATIVE_PATH="krb5-{{version}}/src" \
 bash install-from-source.sh --without-keyutils --without-system-verto --without-libedit --disable-static

# libxml & libxslt for lxml
RUN \
 DOWNLOAD_URL="https://download.gnome.org/sources/libxml2/2.12/libxml2-{{version}}.tar.xz" \
 VERSION="2.12.6" \
 SHA256="889c593a881a3db5fdd96cc9318c87df34eb648edfc458272ad46fd607353fbb" \
 RELATIVE_PATH="libxml2-{{version}}" \
 bash install-from-source.sh \
 --without-iconv \
 --without-python \
 --without-icu \
 --without-debug \
 --without-mem-debug \
 --without-run-debug \
 --without-legacy \
 --without-catalog \
 --without-docbook \
 --disable-static

RUN \
 DOWNLOAD_URL="https://download.gnome.org/sources/libxslt/1.1/libxslt-{{version}}.tar.xz" \
 VERSION="1.1.39" \
 SHA256="2a20ad621148339b0759c4d4e96719362dee64c9a096dbba625ba053846349f0" \
 RELATIVE_PATH="libxslt-{{version}}" \
 bash install-from-source.sh \
 --without-python \
 --without-crypto \
 --without-profiler \
 --without-debugger \
 --disable-static

# libpq and pg_config as needed by psycopg2
RUN \
 DOWNLOAD_URL="https://ftp.postgresql.org/pub/source/v{{version}}/postgresql-{{version}}.tar.bz2" \
 VERSION="16.0" \
 SHA256="df9e823eb22330444e1d48e52cc65135a652a6fdb3ce325e3f08549339f51b99" \
 RELATIVE_PATH="postgresql-{{version}}" \
 bash install-from-source.sh --without-readline --with-openssl --without-icu
# Add paths to pg_config and to the library
ENV PATH="/usr/local/pgsql/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/pgsql/lib/:${LD_LIBRARY_PATH}"

# odbc for pyodbc
RUN \
 DOWNLOAD_URL="https://www.unixodbc.org/unixODBC-{{version}}.tar.gz" \
 VERSION="2.3.9" \
 SHA256="52833eac3d681c8b0c9a5a65f2ebd745b3a964f208fc748f977e44015a31b207" \
 RELATIVE_PATH=unixODBC-{{version}} \
 bash install-from-source.sh --disable-readline --with-included-ltdl --enable-ltdl-install \
 # This is the folder where unixODBC searches for driver config and where we ask customers to copy their config to
 --sysconfdir=/opt/datadog-agent/embedded/etc

# Dependencies needed to build librdkafka (and thus, confluent-kafka) with kerberos support
RUN \
 DOWNLOAD_URL="https://github.com/LMDB/lmdb/archive/LMDB_{{version}}.tar.gz" \
 VERSION="0.9.29" \
 SHA256="22054926b426c66d8f2bc22071365df6e35f3aacf19ad943bc6167d4cae3bebb" \
 RELATIVE_PATH="lmdb-LMDB_{{version}}/libraries/liblmdb" \
 # No ./configure, use a NOOP
 CONFIGURE_SCRIPT="true" \
 bash install-from-source.sh
RUN \
 DOWNLOAD_URL="https://mirrors.edge.kernel.org/pub/linux/kernel/people/tytso/e2fsprogs/v{{version}}/e2fsprogs-{{version}}.tar.gz" \
 VERSION="1.47.0" \
 SHA256="0b4fe723d779b0927fb83c9ae709bc7b40f66d7df36433bef143e41c54257084" \
 RELATIVE_PATH="e2fsprogs-{{version}}" \
 bash install-from-source.sh --enable-elf-shlibs
RUN \
 # Add -fPIC to let librdkafka link against it statically
 CFLAGS="${CFLAGS} -fPIC" \
 # Explicitly ask the linker to use gssapi_krb5, otherwise static compilation fails
 LDFLAGS="${LDFLAGS} -L/usr/local/lib -lgssapi_krb5" \
 DOWNLOAD_URL="https://github.com/cyrusimap/cyrus-sasl/releases/download/cyrus-sasl-{{version}}/cyrus-sasl-{{version}}.tar.gz" \
 VERSION="2.1.28" \
 SHA256="7ccfc6abd01ed67c1a0924b353e526f1b766b21f42d4562ee635a8ebfc5bb38c" \
 RELATIVE_PATH="cyrus-sasl-{{version}}" \
 bash install-from-source.sh --with-dblib=lmdb --enable-gssapi=/usr/local \
    --enable-static --disable-shared
# curl
RUN \
 DOWNLOAD_URL="https://curl.haxx.se/download/curl-{{version}}.tar.gz" \
 VERSION="8.14.0" \
 SHA256="5eba3ec2d898919a4b34cc89ace0068d9d09420b8643f265aebac59f8d126203" \
 RELATIVE_PATH="curl-{{version}}" \
  bash install-from-source.sh \
    --disable-manual \
    --disable-debug \
    --enable-optimize \
    --disable-static \
    --disable-ldap \
    --disable-ldaps \
    --disable-rtsp \
    --enable-proxy \
    --disable-dependency-tracking \
    --enable-ipv6 \
    --without-libidn \
    --without-gnutls \
    --without-librtmp \
    --without-libssh2 \
    --without-libpsl \
    --with-ssl=/usr/local \
 && rm /usr/local/bin/curl

# Environment variables to help openssl crate find OpenSSL
ENV OPENSSL_LIB_DIR="/usr/local/lib64"
ENV OPENSSL_INCLUDE_DIR="/usr/local/include"

# Set up runner
COPY runner_dependencies.txt /runner_dependencies.txt
RUN python3 -m pip install --no-warn-script-location -r /runner_dependencies.txt

COPY build_script.sh /build_script.sh
ENV DD_BUILD_COMMAND="bash /build_script.sh"

ENV MANYLINUX_POLICY="manylinux2014_x86_64"

ENTRYPOINT ["python3", "/home/scripts/build_wheels.py"]
