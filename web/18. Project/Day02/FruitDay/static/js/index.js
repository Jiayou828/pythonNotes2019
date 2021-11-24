window.onload = function (){
	/*-----------下拉菜单---------*/

	//1. 获取元素节点
	var currentAddr = document.getElementsByClassName('currentAddress')[0];
	var select = document.getElementsByClassName('select')[0];
	
	//获取内层列表中地址项
	var address = select.children;
	//为每一项添加点击事件
	for(var i = 0; i < address.length; i ++){
		address[i].onclick = function(){
			//传值
			currentAddr.innerHTML = this.innerHTML;
		};
	}

	/*-----------图片轮播-----------*/
	//1. 获取图片数组
	//2. 定时器实现图片切换
	//3. 图片切换主要切换数组下标，防止数组越界

	var banner = document.getElementsByClassName('wrapper')[0];
	var imgs = banner.children; //图片数组
	var imgNav = document.getElementsByClassName('imgNav')[0];
	var indInfo = imgNav.children; //索引数组
	var imgIndex = 0; //初始下标
	var timer;
	timer = setInterval(autoPlay,1000); //定时器
	function autoPlay(){
		//设置元素隐藏与显示
		imgs[imgIndex].style.display = "none";
		/*
		++ imgIndex;
		if(imgIndex == imgs.length){
			imgIndex = 0;
		}
		*/
		imgIndex = ++ imgIndex == imgs.length ? 0 : imgIndex;

		imgs[imgIndex].style.display = "block";

		for(var i = 0; i < indInfo.length; i ++){
			indInfo[i].style.background = "gray";
		}
		//切换索引 切换背景色
		indInfo[imgIndex].style.background = "red";
	}
	banner.onmouseover = function (){
		//停止定时器
		clearInterval(timer);
	};

	banner.onmouseout = function (){
		timer = setInterval(autoPlay,1000);
	};

};

/**异步向服务器发送请求，检查用户是否处于登录状态*/
function check_login(){
	// 向　/check_login/ 发送异步请求
	$.get('/check_login/',function(data){
		var html = "";
		if(data.loginStatus == 0){
			html += "<a href='/login'>[登录]</a>";
			html += "<a href='/register'>[注册,有惊喜]</a>";
		}else{
			html += "欢迎:"+data.uname;
			html += "<a href='/logout'>退出</a>";
		}
		$("#login").html(html);
	},'json');
}

/**加载所有的商品分类以及商品信息(每个分类取前10个)*/
function loadGoods(){
	$.get('/load_type_goods/',function(data){
		//data 就是响应回来的JSON对象
		console.log(data);
	},'json');
}


$(function(){
	/*调用check_login检查登录状态*/
	check_login();
	/*调用loadGoods函数得到所有的类型和商品*/
	loadGoods();
});