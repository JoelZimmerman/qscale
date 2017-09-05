%module qscale

%{
#include "libqscale.h"

int qscale_is_invalid(qscale_img *img) {
        return (img == NULL);
}
%}
%include "libqscale.h"

int qscale_is_invalid(qscale_img *img);
