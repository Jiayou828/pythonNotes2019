from django.conf.urls import url
from .views import *
urlpatterns = [
  url(r'^login/$',login_views,name='login'),
  url(r'^register/$',register_views,name='reg'),
  url(r'^check_uphone/$',check_uphone_views),
]