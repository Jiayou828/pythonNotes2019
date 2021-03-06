import json

from django.http import HttpResponse, HttpResponseRedirect
from django.shortcuts import render, redirect
from .models import *
from .forms import *

# Create your views here.
#　http://localhost:8000/login
def login_views(request):
  # 判断　get 请求还是　post　请求
  if request.method == 'GET':
    #　获取来访地址，如果没有则设置为/
    url = request.META.get('HTTP_REFERER','/')
    #get请求　－　判断session,判断cookie,登录页
    #先判断session中是否有登录信息
    if 'uid' in request.session and 'uphone' in request.session:
      #有登录信息保存在　session
      # 从哪来，回哪去
      resp = HttpResponseRedirect(url)
      return resp
    else:
      #没有登录信息保存在　session，继续判断cookies中是否有登录信息
      if 'uid' in request.COOKIES and 'uphone' in request.COOKIES:
        #cookies中有登录信息　－　曾经记住过密码
        #将cookies中的信息取出来保存进session，再返回到首页
        uid = request.COOKIES['uid']
        uphone = request.COOKIES['uphone']
        request.session['uid']=uid
        request.session['uphone']=uphone
        # 从哪来，回哪去
        resp = redirect(url)
        return resp
      else:
        #cookies中没有登录信息　－　去往登录页
        form = LoginForm()
        #将来访地址保存进cookies中
        resp = render(request,'login.html',locals())
        resp.set_cookie('url',url)
        return resp
  else:
    #post请求 - 实现登录操作
    #获取手机号和密码
    uphone = request.POST['uphone']
    upwd = request.POST['upwd']
    #判断手机号和密码是否存在(登录是否成功)
    users=User.objects.filter(uphone=uphone,upwd=upwd)
    if users:
      #登录成功：先存进session
      request.session['uid']=users[0].id
      request.session['uphone']=uphone
      #声明响应对象：从哪来回哪去
      url = request.COOKIES.get('url','/')
      resp = redirect(url)
      #将url从cookies中删除出去
      if 'url' in request.COOKIES:
        resp.delete_cookie('url')
      #判断是否要存进cookies
      if 'isSaved' in request.POST:
        expire = 60*60*24*90
        resp.set_cookie('uid',users[0].id,expire)
        resp.set_cookie('uphone',uphone,expire)
      return resp
    else:
      #登录失败
      form = LoginForm()
      return render(request,'login.html',locals())

# http://localhost:8000/register
def register_views(request):
  # 判断是get请求还是post请求，得到用户的请求意图
  if request.method == 'GET':
    return render(request,'register.html')
  else:
    # #先验证手机号在数据库中是否存在
    uphone = request.POST['uphone']
    # users = User.objects.filter(uphone=uphone)
    # if users:
    #   #uphone 已经存在
    #   errMsg = '手机号码已经存在'
    #   return render(request,'register.html',locals())
    #接收数据插入到数据库中
    upwd = request.POST['upwd']
    uname = request.POST['uname']
    uemail = request.POST['uemail']
    user = User()
    user.uphone = uphone
    user.upwd = upwd
    user.uname = uname
    user.uemail = uemail
    user.save()
    #取出user中的id 和 uphone的值保存进session
    request.session['uid']=user.id
    request.session['uphone']=user.uphone
    return HttpResponse('注册成功')


# 检查手机号是否已经被注册过
def check_uphone_views(request):
  # 接收前端传递过来的数据 - uphone
  uphone = request.GET['uphone']
  users = User.objects.filter(uphone = uphone)
  if users:
    status = 1
    msg = '手机号码已经存在'
  else:
    status = 0
    msg = '通过'

  dic = {
    'status':status,
    'msg' : msg,
  }
  return HttpResponse(json.dumps(dic))



def index_views(request):
  return render(request,'index.html')

# 检查　session 中是否有登录信息，如果有获取对应数据的uname值
def check_login_views(request):
  if 'uid' in request.session and 'uphone' in request.session:
    loginStatus = 1
    #通过uid的值获取对应的uname
    id = request.session['uid']
    uname=User.objects.get(id=id).uname
    dic = {
      'loginStatus':loginStatus,
      'uname':uname
    }
    return HttpResponse(json.dumps(dic))
  else:
    dic = {
      'loginStatus':0
    }
    return HttpResponse(json.dumps(dic))

#退出
def logout_views(request):
  #判断session中是否有登录信息，有的话则清除
  if 'uid' in request.session and 'uphone' in request.session:
    del request.session['uid']
    del request.session['uphone']
    #构建响应对象：哪发的退出请求，则返回到哪去
    url=request.META.get('HTTP_REFERER','/')
    resp = HttpResponseRedirect(url)
    #判断cookies中是否有登录信息，有的话，则删除
    if 'uid' in request.COOKIES and 'uphone' in request.COOKIES:
      resp.delete_cookie('uid')
      resp.delete_cookie('uphone')
    return resp
  return redirect('/')


def test_views(request):
  return render(request,'test.html')



